"""Based on https://github.com/lincc-frameworks/nested-pandas/issues/89"""

import nested_pandas as npd
import numpy as np


def test_issue89():
    """Check that code snippet from issue 89 works as expected

    https://github.com/lincc-frameworks/nested-pandas/issues/89
    """

    # Load some ZTF data
    catalogs_dir = "https://epyc.astro.washington.edu/~lincc-frameworks/half_degree_surveys/ztf/"

    object_ndf = npd.read_parquet(
        f"{catalogs_dir}/ztf_object/Norder=3/Dir=0/Npix=432.parquet",
        columns=["ra", "dec", "ps1_objid"],
    ).set_index("ps1_objid")

    source_ndf = npd.read_parquet(
        f"{catalogs_dir}/ztf_source/Norder=6/Dir=20000/Npix=27711.parquet",
        columns=["mjd", "mag", "magerr", "band", "ps1_objid", "catflags"],
    ).set_index("ps1_objid")

    object_ndf = object_ndf.add_nested(source_ndf, "ztf_source")

    nf = object_ndf
    nf.reduce(np.mean, "ztf_source.mjd")
