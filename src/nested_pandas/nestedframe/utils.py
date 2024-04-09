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
