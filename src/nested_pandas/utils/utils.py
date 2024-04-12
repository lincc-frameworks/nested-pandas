import pandas as pd

from nested_pandas import NestedFrame


def count_nested(df, nested, by=None, join=True) -> NestedFrame:
    """Counts the number of rows of a nested dataframe.

    Parameters
    ----------
    df: NestedFrame
        A NestedFrame that contains the desired `nested` series
        to count.
    nested: 'str'
        The label of the nested series to count.
    by: 'str', optional
        Specifies a column within nested to count by, returning
        a count for each unique value in `by`.
    join: bool, optional
        Join the output count columns to df and return df, otherwise
        just return a NestedFrame containing only the count columns.

    Returns
    -------
    NestedFrame
    """

    if by is None:
        counts = df[nested].apply(lambda x: len(x)).rename(f"n_{nested}")
    else:
        counts = df[nested].apply(lambda x: x[by].value_counts())
        counts = counts.rename(columns={colname: f"n_{nested}_{colname}" for colname in counts.columns})
    if join:
        return df.join(counts)
    # else just return the counts NestedFrame
    if isinstance(counts, pd.Series):  # for by=None, which returns a Series
        counts = NestedFrame(counts.to_frame())
    return counts
