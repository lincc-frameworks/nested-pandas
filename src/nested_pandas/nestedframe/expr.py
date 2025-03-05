"""Utilities used by NestedFrame.query() and .eval()"""

# typing.Self and "|" union syntax don't exist in Python 3.9
from __future__ import annotations

import ast
import re
from typing import TYPE_CHECKING

import pandas as pd
from pandas.core.computation import ops
from pandas.core.computation.expr import PARSERS, PandasExprVisitor
from pandas.core.computation.parsing import clean_column_name

# Avoid cyclic import
if TYPE_CHECKING:
    from nested_pandas import NestedFrame

# Used to identify backtick-protected names in the expressions
# used in NestedFrame.eval() and NestedFrame.query().
_backtick_protected_names = re.compile(r"`[^`]+`", re.MULTILINE)


class NestedPandasExprVisitor(PandasExprVisitor):
    """
    Custom expression visitor for NestedFrame evaluations, which may assign to
    nested columns.
    """

    def visit_Assign(self, node, **kwargs):  # noqa: N802
        """
        Visit an assignment node, which may assign to a nested column.
        """
        if not isinstance(node.targets[0], ast.Attribute):
            # If the target is not an attribute, then it's a simple assignment as usual
            return super().visit_Assign(node)
        target = node.targets[0]
        if not isinstance(target.value, ast.Name):
            raise ValueError("Assignments to nested columns must be of the form `nested.col = ...`")
        # target.value.id will be the name of the nest, target.attr is the column name.
        # Describing the proper target for the assigner is enough for both overwrite and
        # creation of new columns.  The assigner will be a string like "nested.col".
        # This works both for the creation of new nest members and new nests.
        self.assigner = f"{target.value.id}.{target.attr}"
        # Continue visiting.
        return self.visit(node.value, **kwargs)


PARSERS["nested-pandas"] = NestedPandasExprVisitor


class _SeriesFromNest(pd.Series):
    """
    Series that were unpacked from a nest.
    """

    _metadata = ["nest_name", "flat_nest"]

    @property
    def _constructor(self) -> Self:  # type: ignore[name-defined] # noqa: F821
        return _SeriesFromNest

    @property
    def _constructor_expanddim(self) -> Self:  # type: ignore[name-defined] # noqa: F821
        # Avoid cyclic import
        from nested_pandas import NestedFrame

        return NestedFrame

    # https://pandas.pydata.org/docs/development/extending.html#arithmetic-with-3rd-party-types
    # The __pandas_priority__ of Series is 3000, so give _SeriesFromNest a
    # higher priority, so that binary operations involving this class and
    # Series produce instances of this class, preserving the type and origin.
    __pandas_priority__ = 3500


class _NestResolver(dict):
    """
    Used by NestedFrame.eval to resolve the names of nests at the top level.
    While the resolver is normally a dictionary, with values that are fixed
    upon entering evaluation, this object needs to be dynamic so that it can
    support multi-line expressions, where new nests may be created during
    evaluation.
    """

    def __init__(self, outer: NestedFrame):
        self._outer = outer
        super().__init__()
        # Pre-load the field resolvers for all columns which are known at present.
        for column in outer.nested_columns:
            self._initialize_column_resolver(column, outer)

    def _initialize_column_resolver(self, column: str, outer: NestedFrame):
        """
        Initialize a resolver for the given nested column, and also an alias
        for it, in the case of column names that have spaces or are otherwise
        not identifier-like.
        """
        super().__setitem__(column, _NestedFieldResolver(column, outer))
        clean_id = clean_column_name(column)
        # And once more for the cleaned name, if it's different.
        # This allows us to capture references to it from the Pandas evaluator.
        if clean_id != column:
            super().__setitem__(clean_id, _NestedFieldResolver(column, outer))

    def __contains__(self, item):
        top_nest = item if "." not in item else item.split(".")[0].strip()
        return top_nest in self._outer.nested_columns

    def __getitem__(self, item):
        top_nest = item if "." not in item else item.split(".")[0].strip()
        if not super().__contains__(top_nest):
            if top_nest not in self._outer.nested_columns:
                raise KeyError(f"Unknown nest {top_nest}")
            self._initialize_column_resolver(top_nest, self._outer)
        return super().__getitem__(top_nest)

    def __setitem__(self, item, _):
        # Called to update the resolver with intermediate values.
        # The important point is to intercept the call so that the evaluator
        # does not create any new resolvers on the fly.  We do NOT want to
        # store the given value, since the resolver does lazy-loading.
        # What we DO want to do, however, is to invalidate the cache for
        # any field resolver for a given nest that is receiving an assignment.
        # Since the resolvers are created as-needed in __getitem__, all we need
        # to do is delete them from the local cache when this pattern is detected.
        if "." in item:
            top_nest = item.split(".")[0].strip()
            if top_nest in self._outer.nested_columns and super().__contains__(top_nest):
                del self[top_nest]  # force re-creation in __setitem__


class _NestedFieldResolver:
    """
    Used by NestedFrame.eval to resolve the names of fields in nested columns when
    encountered in expressions, interpreting __getattr__ in terms of a
    specific nest.
    """

    def __init__(self, nest_name: str, outer: NestedFrame):
        self._nest_name = nest_name
        # Save the outer frame with an eye toward repacking.
        self._outer = outer
        # Flattened only once for every access of this particular nest
        # within the expression.
        self._flat_nest = outer[nest_name].nest.to_flat()
        # Save aliases to any columns that are not identifier-like.
        # If our given frame has aliases for identifiers, use these instead
        # of generating our own.
        self._aliases = getattr(outer, "_aliases", None)
        if self._aliases is None:
            self._aliases = {}
            for column in self._flat_nest.columns:
                clean_id = clean_column_name(column)
                if clean_id != column:
                    self._aliases[clean_id] = column

    def __getattr__(self, item_name: str):
        if self._aliases:
            item_name = self._aliases.get(item_name, item_name)
        if item_name in self._flat_nest:
            result = _SeriesFromNest(self._flat_nest[item_name])
            # Assigning these properties directly in order to avoid any complication
            # or interference with the inherited pd.Series constructor.
            result.nest_name = self._nest_name
            result.flat_nest = self._flat_nest
            return result
        raise AttributeError(f"No attribute {item_name}")


def _subexprs_by_nest(parents: list, node) -> dict[str, list]:
    """
    Given an expression which contains references to both base and nested
    columns, return a dictionary of the sub-expressions that should be
    evaluated independently, keyed by nesting context.

    The key of the dictionary is the name of the nested column, and will
    be a blank string in the case of base columns.  The value is a list
    of the parent nodes that lead to sub-expressions that can be evaluated
    successfully.

    While this is not in use today for automatically splitting expressions,
    it can be used to detect whether an expression is suitably structured
    for evaluation: the returned dictionary should have a single key.
    """
    if isinstance(node, ops.Term) and not isinstance(node, ops.Constant):
        if isinstance(node.value, _SeriesFromNest):
            return {node.value.nest_name: parents}
        return {getattr(node, "upper_name", ""): parents}
    if not isinstance(node, ops.Op):
        return {}
    sources = [getattr(node, "lhs", None), getattr(node, "rhs", None)]
    result: dict[str, list] = {}
    for source in sources:
        child = _subexprs_by_nest(parents, source)
        for k, v in child.items():
            result.setdefault(k, []).append(v)
    # After a complete traversal across sources, check for any necessary splits.
    # If it's homogenous, move the split-node up the tree.
    if len(result) == 1:
        # Let the record of each parent node drift up the tree,
        # and merge the subtrees into a single node, since by definition,
        # this node is homogeneous over all of its children, and can
        # be evaluated in a single step.
        result = {k: [node] for k in result}
    # If the result is either empty or has more than one key, leave the result
    # alone.  Each key represents a different nest (with a blank string for the base),
    # and the value is the highest point in the expression tree where the expression
    # was still within a single nest.
    return result


def _identify_aliases(expr: str) -> tuple[str, dict[str, str]]:
    """
    Given an expression string, identify backtick-quoted names
    and replace them with cleaned names, returning the cleaned
    expression and a dictionary of aliases, where the keys are
    clean aliases to the original names.
    """
    aliases = {}

    def sub_and_alias(match):
        original = match.group(0)[1:-1]  # remove backticks
        alias = clean_column_name(original)
        if alias != original:
            aliases[alias] = original
        return alias

    return _backtick_protected_names.sub(sub_and_alias, expr), aliases
