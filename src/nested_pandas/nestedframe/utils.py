# typing.Self and "|" union syntax don't exist in Python 3.9
from __future__ import annotations

import ast


def _subexprs_by_nest(parents: list[ast.expr], node: ast.expr | None) -> dict[str, list]:
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
    if not isinstance(node, ast.expr):
        return {}
    if isinstance(node, ast.Name):
        return {"": parents}
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return {node.value.id: parents}
    sources = (
        [getattr(node, "left", None), getattr(node, "right", None)]
        + getattr(node, "values", [])
        + getattr(node, "comparators", [])
    )
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


def extract_nest_names(expr: str) -> set[str]:
    """
    Given a string expression, parse it and visit the resulting AST, surfacing
    the nesting types.  The purpose is to identify expressions that attempt
    to mix base and nested columns, or columns from two different nests.
    """
    expr_tree = ast.parse(expr, mode="eval").body
    separable = _subexprs_by_nest([], expr_tree)
    return set(separable.keys())
