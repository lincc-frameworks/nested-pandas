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


def check_expr_nesting(expr: str) -> set[NestingType]:
    """
    Given a string expression, parse it and visit the resulting AST, surfacing
    the nesting types.  The purpose is to identify expressions that attempt
    to mix base and nested columns, which will need to be handled specially.
    """
    expr_tree = ast.parse(expr, mode="eval").body
    return _expr_nesting_type(expr_tree)
