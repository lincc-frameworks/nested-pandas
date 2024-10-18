# typing.Self and "|" union syntax don't exist in Python 3.9
from __future__ import annotations

import ast
from enum import Enum


class NestingType(Enum):
    """Types of sub-expressions possible in a NestedFrame string expression."""

    BASE = "base"
    NESTED = "nested"


def _expr_nesting_type(node: ast.expr | None) -> set[NestingType]:
    if not isinstance(node, ast.expr):
        return set()
    if isinstance(node, ast.Name):
        return {NestingType.BASE}
    if isinstance(node, ast.Attribute):
        return {NestingType.NESTED}
    sources = (
        [getattr(node, "left", None), getattr(node, "right", None)]
        + getattr(node, "values", [])
        + getattr(node, "comparators", [])
    )
    result: set[NestingType] = set()
    for s in sources:
        result.update(_expr_nesting_type(s))
    return result


def _actionable_splits(parents: list[ast.expr], node: ast.expr | None) -> dict[str, list]:
    """
    Given an expression which contains references to both base and nested columns,
    return a list of the sub-expressions that should be evaluated independently.
    The goal is to enable the ability to reduce the nested sub-expressions without
    broadcasting.  Thus, an expression like "a > 0.5 and nested.b > 50" should be
    split into ["a > 0.5", "nested.b > 50"].  Moreover, given that the logical
    operator is "and", all the rows that fail the first condition can be discarded,
    and the nest must be further reduced by the second condition.  If, on the other
    hand, the logical operator is "or", having the base condition succeed means
    that we never have to evaluate the nested condition.

    Ideally, then, this function should produce a series of steps that can be
    evaluated in sequence, as if it were a chain of query() calls.  This requires
    deep analysis of the logical expression.

    "a > 0.5 and (nested.b > 50 or nested.c > 0) or (b < 2 and nested.d > 100)"
    is an example of a tough one.  There are, broadly, two optimization goals:
    save CPU time, and save memory.  The second rates higher; we never want to
    induce broadcasting.  If we don't care about accidentally calculating
    throwaway partial results, we can narrow the nested columns first.  But
    note the "b > 2" condition in the final clause; this means that we can't
    just reduce the nested columns to those that pass the first condition.
    Essentially, we need to rewrite the entire expression so that all the
    nested checks are on one side and the base checks are on the other:

    "a > 0.5 and (b < 2 and nested.d > 100) or (nested.b > 50 or nested.c > 0)"

    What's nice about the ast parser is that when it's all "or" at one level,
    it's clear as such; same with "and".

    Mixed is difficult because we don't want the Pandas machinery to blow memory.
    So for "(b < 2 and nested.d > 100)", we want to query base first, because
    that will shrink memory fastest.  Then apply "(nested.d > 100)" to the result.

    Wait, is it really that simple?  Just evaluate the base first, then the nested?
    I don't think so.

    The sub-expressions should also be partitioned or split by nest name.
    "n1.a > 2 and n2.b > 3" should be split into ["n1.a > 2", "n2.b > 3"].
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
    for s in sources:
        child = _actionable_splits(parents, s)
        for k, v in child.items():
            result[k] = result.get(k, []) + v
    # After a complete traversal across sources, check for any necessary splits.
    # If it's homogenous, move the split-node up the tree.
    if len(result) == 1:
        # Let the record of each parent node drift up the tree,
        # and merge the subtrees into a single node, since by definition,
        # this node is homogeneous over all of its children, and can
        # be evaluated in a single step.
        result = {k: [node] for k in result}
    else:
        # At this point, we need to split the expression.  The idea here is that
        # we want a succession of efficient queries, each of which will produce
        # a subset of either the base or the nested columns.
        pass
    return result


def check_expr_nesting(expr: str) -> set[NestingType]:
    """
    Given a string expression, parse it and visit the resulting AST, surfacing
    the nesting types.  The purpose is to identify expressions that attempt
    to mix base and nested columns, which will need to be handled specially.
    """
    expr_tree = ast.parse(expr, mode="eval").body
    return _expr_nesting_type(expr_tree)
