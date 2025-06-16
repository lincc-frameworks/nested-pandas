import numpy as np
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

    Examples
    --------

    >>> import pandas as pd
    >>> # Show all columns
    >>> pd.set_option("display.width", 200)
    >>> pd.set_option("display.max_columns", None)
    >>> from nested_pandas.datasets.generation import generate_data
    >>> nf = generate_data(5, 10, seed=1)

    >>> from nested_pandas.utils import count_nested
    >>> count_nested(nf, "nested")
              a         b                                             nested  n_nested
    0  0.417022  0.184677  [{t: 8.38389, flux: 10.233443, band: 'g'}; …] ...        10
    1  0.720324  0.372520  [{t: 13.70439, flux: 41.405599, band: 'g'}; …]...        10
    2  0.000114  0.691121  [{t: 4.089045, flux: 69.440016, band: 'g'}; …]...        10
    3  0.302333  0.793535  [{t: 17.562349, flux: 41.417927, band: 'g'}; …...        10
    4  0.146756  1.077633  [{t: 0.547752, flux: 4.995346, band: 'r'}; …] ...        10

    `count_nested` also allows counting by a given subcolumn, for example we
    can count by "band" label:

    >>> # join=False, allows the result to be kept separate from the original nf
    >>> count_nested(nf, "nested", by="band", join=False)
       n_nested_g  n_nested_r
    0           8           2
    1           5           5
    2           5           5
    3           6           4
    4           6           4
    """

    if by is None:
        counts = pd.Series(df[nested].nest.list_lengths, name=f"n_{nested}", index=df.index)
    else:
        counts = df.reduce(
            lambda x: dict(zip(*np.unique(x, return_counts=True), strict=False)), f"{nested}.{by}"
        )
        counts = counts.rename(columns={colname: f"n_{nested}_{colname}" for colname in counts.columns})
        counts = counts.reindex(sorted(counts.columns), axis=1)
    if join:
        return df.join(counts)
    # else just return the counts NestedFrame
    if isinstance(counts, pd.Series):  # for by=None, which returns a Series
        counts = NestedFrame(counts.to_frame())
    return counts
