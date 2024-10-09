import ast
from enum import Enum


def _ensure_spacing(expr) -> str:
    """Ensure that an eval string has spacing"""
    single_val_operators = {"+", "-", "*", "/", "%", ">", "<", "|", "&", "~", "="}  # omit "(" and ")"
    check_for_doubles = {"=", "/", "*", ">", "<"}
    double_val_operators = {"==", "//", "**", ">=", "<="}
    expr_list = expr

    i = 0
    spaced_expr = ""
    while i < len(expr_list):
        if expr_list[i] not in single_val_operators:
            spaced_expr += expr_list[i]
        else:
            if expr_list[i] in check_for_doubles:
                if "".join(expr_list[i : i + 2]) in double_val_operators:
                    if spaced_expr[-1] != " ":
                        spaced_expr += " "
                    spaced_expr += expr_list[i : i + 2]
                    if expr_list[i + 2] != " ":
                        spaced_expr += " "
                    i += 1  # skip ahead an extra time
                else:
                    if spaced_expr[-1] != " ":
                        spaced_expr += " "
                    spaced_expr += expr_list[i]
                    if expr_list[i + 1] != " ":
                        spaced_expr += " "
            else:
                if spaced_expr[-1] != " ":
                    spaced_expr += " "
                spaced_expr += expr_list[i]
                if expr_list[i + 1] != " ":
                    spaced_expr += " "
        i += 1
    return spaced_expr


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
    return set(_expr_nesting_type(expr_tree))
