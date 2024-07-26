import pandas as pd

from nested_pandas import NestedFrame


def count_nested(df, nested, by=None, join=True) -> NestedFrame:
    """Counts the number of rows of a nested dataframe.

    #TODO: Does not work when any nested dataframes are empty (NaN)

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
        field_to_len = df[nested].nest.fields[0]
        counts = df[nested].nest.to_lists().apply(lambda x: len(x[field_to_len]), axis=1)
        counts.name = f"n_{nested}"  # update name directly (rename causes issues downstream)
    else:
        # this may be able to be sped up using tolists() as well
        counts = df[nested].apply(lambda x: x[by].value_counts(sort=False))
        counts = counts.rename(columns={colname: f"n_{nested}_{colname}" for colname in counts.columns})
        counts = counts.reindex(sorted(counts.columns), axis=1)
    if join:
        return df.join(counts)
    # else just return the counts NestedFrame
    if isinstance(counts, pd.Series):  # for by=None, which returns a Series
        counts = NestedFrame(counts.to_frame())
    return counts
