"""Two sample benchmarks to compute runtime and memory usage.

For more information on writing benchmarks:
https://asv.readthedocs.io/en/stable/writing_benchmarks.html."""

import os
import platform
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
from upath import UPath

from nested_pandas import NestedDtype, NestedFrame, datasets, read_parquet
from nested_pandas.utils import count_nested


class AssignSingleDfToNestedSeries:
    """Benchmark the performance of changing a single nested series element"""

    n_objects = 10_000
    n_sources = 100
    new_df: pd.DataFrame
    series: pd.Series

    def setup(self):
        """Set up the benchmark environment."""
        self.new_df = pd.DataFrame(
            {
                "time": np.arange(self.n_sources, dtype=np.float64),
                "flux": np.linspace(0, 1, self.n_sources),
                "band": np.full_like("lsstg", self.n_sources),
            }
        )
        original_df = pd.DataFrame(
            {
                "time": np.linspace(0, 1, self.n_sources),
                "flux": np.arange(self.n_sources, dtype=np.float64),
                "band": np.full_like("sdssu", self.n_sources),
            }
        )
        self.series = pd.Series(
            [original_df] * self.n_objects,
            # When we had NestedExtentionArray inheriting ArrowExtentionArray, it sorted the fields, so we
            # need to order by field name here for backwards compatibility.
            dtype=NestedDtype.from_columns({"band": pa.string(), "flux": pa.float64(), "time": pa.float64()}),
        )

    def run(self):
        """Run the benchmark."""
        self.series[self.n_objects // 2] = self.new_df

    def time_run(self):
        """Benchmark the runtime of changing a single nested series element."""
        self.run()

    def peakmem_run(self):
        """Benchmark the memory usage of changing a single nested series element."""
        self.run()


class ReassignHalfOfNestedSeries:
    """Benchmark the performance of changing a lot of nested series elements"""

    n_objects = 10_000
    n_sources = 100
    series: pd.Series
    new_series: pd.Series

    def setup(self):
        """Set up the benchmark environment."""
        # When we had NestedExtentionArray inheriting ArrowExtentionArray, it sorted the fields, so we need to
        # order by field name here for backwards compatibility.
        dtype = NestedDtype.from_columns({"band": pa.string(), "flux": pa.float64(), "time": pa.float64()})
        original_df = pd.DataFrame(
            {
                "time": np.linspace(0, 1, self.n_sources),
                "flux": np.arange(self.n_sources, dtype=np.float64),
                "band": np.full_like("sdssu", self.n_sources),
            }
        )
        self.series = pd.Series(
            [original_df] * self.n_objects,
            dtype=dtype,
        )

        new_df = pd.DataFrame(
            {
                "time": np.arange(self.n_sources, dtype=np.float64),
                "flux": np.linspace(0, 1, self.n_sources),
                "band": np.full_like("lsstg", self.n_sources),
            }
        )
        self.new_series = pd.Series([new_df] * (self.n_objects // 2), dtype=dtype)

    def run(self):
        """Run the benchmark."""
        self.series[::2] = self.new_series

    def time_run(self):
        """Benchmark the runtime of changing a single nested series element."""
        self.run()

    def peakmem_run(self):
        """Benchmark the memory usage of changing a single nested series element."""
        self.run()


class NestedFrameAddNested:
    """Benchmark the NestedFrame.join_nested function"""

    n_base = 100
    layer_size = 1000
    base_nf: NestedFrame
    layer_nf: NestedFrame

    def setup(self):
        """Set up the benchmark environment"""
        # use provided seed, "None" acts as if no seed is provided
        randomstate = np.random.RandomState(seed=1)

        # Generate base data
        base_data = {"a": randomstate.random(self.n_base), "b": randomstate.random(self.n_base) * 2}
        self.base_nf = NestedFrame(data=base_data)

        layer_data = {
            "t": randomstate.random(self.layer_size * self.n_base) * 20,
            "flux": randomstate.random(self.layer_size * self.n_base) * 100,
            "band": randomstate.choice(["r", "g"], size=self.layer_size * self.n_base),
            "index": np.arange(self.layer_size * self.n_base) % self.n_base,
        }
        self.layer_nf = NestedFrame(data=layer_data).set_index("index")

    def run(self):
        """Run the benchmark."""
        self.base_nf.join_nested(self.layer_nf, "nested")

    def time_run(self):
        """Benchmark the runtime of adding a nested layer"""
        self.run()

    def peakmem_run(self):
        """Benchmark the memory usage of adding a nested layer"""
        self.run()


class NestedFrameMapRows:
    """Benchmark the NestedFrame.map_rows function"""

    n_base = 100
    n_nested = 1000
    nf: NestedFrame

    def setup(self):
        """Set up the benchmark environment"""
        self.nf = datasets.generate_data(self.n_base, self.n_nested)

    def run(self):
        """Run the benchmark."""
        self.nf.map_rows(np.mean, columns=["nested.flux"], row_container="args")

    def time_run(self):
        """Benchmark the runtime of applying the map_rows function"""
        self.run()

    def peakmem_run(self):
        """Benchmark the memory usage of applying the map_rows function"""
        self.run()


class NestedFrameQuery:
    """Benchmark the NestedFrame.query function"""

    n_base = 100
    n_nested = 1000
    nf: NestedFrame

    def setup(self):
        """Set up the benchmark environment"""
        self.nf = datasets.generate_data(self.n_base, self.n_nested)

    def run(self):
        """Run the benchmark."""

        # Apply nested layer query
        self.nf = self.nf.query("nested.band == 'g'")

    def time_run(self):
        """Benchmark the runtime of applying the two queries"""
        self.run()

    def peakmem_run(self):
        """Benchmark the memory usage of applying the two queries"""
        self.run()


class CountNestedBy:
    """Benchmark count_nested(nf, by=...)"""

    n_base = 1000
    n_nested = 300
    nf: NestedFrame

    def setup(self):
        """Set up the benchmark environment"""
        self.nf = datasets.generate_data(self.n_base, self.n_nested)

    def run(self):
        """Run the benchmark."""
        _ = count_nested(self.nf, nested="nested", by="band")

    def time_run(self):
        """Benchmark the runtime of count_nested(nf, by=...)"""
        self.run()

    def peakmem_run(self):
        """Benchmark the memory usage of count_nested(nf, by=...)"""
        self.run()


class ReadFewColumnsS3:
    """Benchmark read_parquet("s3://", columns=[...])"""

    # Replace with string when S3 access is reimplemented
    path = UPath(
        "s3://ipac-irsa-ztf/contributed/dr23/lc/hats/ztf_dr23_lc-hats/dataset/Norder=3/Dir=0/Npix=257/part0.snappy.parquet",
        anon=True,
    )
    columns = ["_healpix_29", "lightcurve.mag"]

    def run(self):
        """Run the benchmark."""
        _ = read_parquet(self.path, columns=self.columns, is_dir=False)

    def time_run(self):
        """Benchmark the runtime of read_parquet(self.path, columns=self.columns)"""
        self.run()

    def peakmem_run(self):
        """Benchmark the memory usage of read_parquet(self.path, columns=self.columns)"""
        self.run()


class ReadFewColumnsHTTPS:
    """Benchmark read_parquet("https://", columns=[...])"""

    path = "https://data.lsdb.io/hats/gaia_dr3/gaia/dataset/Norder=2/Dir=0/Npix=0.parquet"
    columns = ["_healpix_29", "ra", "astrometric_primary_flag"]

    def run(self):
        """Run the benchmark."""
        _ = read_parquet(self.path, columns=self.columns, is_dir=False)

    def time_run(self):
        """Benchmark the runtime of read_parquet(self.path, columns=self.columns)"""
        self.run()

    def peakmem_run(self):
        """Benchmark the memory usage of read_parquet(self.path, columns=self.columns)"""
        self.run()


class ReadFewColumnsTmpfs:
    """Benchmark read_parquet("<local path>", columns=[...]) on a tmpfs-cached file.

    Falls back to a disk temp dir off-Linux, for local testing.
    """

    # May be required to download the original file
    timeout = 300

    columns = ["_healpix_29", "ra", "stetsonk"]
    original_path = (
        UPath("s3://ipac-irsa-ztf/ztf/enhanced/dr24/objects/hats/ztf_dr24_objects-hats/", anon=True)
        / "dataset/Norder=4/Dir=0/Npix=98/part0.snappy.parquet"
    )
    local_path: UPath

    def setup_cache(self) -> bytes:
        """Download the source parquet file once; cached and reused across repeats."""
        return self.original_path.read_bytes()

    def setup(self, data):
        """Write the cached bytes to a tmpfs (or temp dir) file before each repeat."""
        tmpfs_dir = "/dev/shm" if platform.system() == "Linux" else tempfile.gettempdir()
        self.local_path = UPath(tmpfs_dir) / "nested_pandas_bench_read_few_columns.parquet"
        self.local_path.write_bytes(data)

    def teardown(self, _data):
        """Remove the tmpfs file after each repeat."""
        self.local_path.unlink(missing_ok=True)

    def run(self):
        """Run the benchmark."""
        _ = read_parquet(
            self.local_path,
            columns=self.columns,
            is_dir=False,
            use_pandas_metadata=False,
        )

    def time_run(self, _data):
        """Benchmark the runtime of read_parquet on the tmpfs file."""
        self.run()

    def peakmem_run(self, _data):
        """Benchmark the memory usage of read_parquet on the tmpfs file."""
        self.run()


class ReadFewColumnsColdDisk:
    """Benchmark read_parquet("<local path>", columns=[...]) with a cold page cache.

    Cache eviction works on Linux only.
    """

    # May be required to download the original file and write it to disk
    timeout = 600

    # A single run, no warm-up: a repeated read would hit the page cache
    number = 1
    repeat = 1
    rounds = 1
    warmup_time = 0.0
    min_run_count = 1
    processes = 1

    columns = ["_healpix_29", "objra", "objdec", "lightcurve.mag"]
    original_path = (
        UPath("s3://ipac-irsa-ztf/ztf/enhanced/dr24/lc/hats/ztf_dr24_lc-hats/", anon=True)
        / "dataset/Norder=5/Dir=10000/Npix=12240/part0.snappy.parquet"
    )
    local_path: UPath

    @staticmethod
    def evict_page_cache(path) -> None:
        """Drop the page cache for a single file, if the platform allows it."""
        if not hasattr(os, "posix_fadvise"):
            return
        os.sync()
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)

    def setup_cache(self) -> bytes:
        """Download the source parquet file once; cached and reused across repeats."""
        return self.original_path.read_bytes()

    def setup(self, data):
        """Write the file to a disk-backed temp dir and evict it from the page cache."""
        self.local_path = UPath(tempfile.gettempdir()) / "nested_pandas_bench_cold_read.parquet"
        self.local_path.write_bytes(data)
        self.evict_page_cache(self.local_path)

    def teardown(self, _data):
        """Remove the temporary file after each repeat."""
        self.local_path.unlink(missing_ok=True)

    def run(self):
        """Run the benchmark."""
        _ = read_parquet(
            self.local_path,
            columns=self.columns,
            is_dir=False,
            use_pandas_metadata=False,
        )

    def time_run(self, _data):
        """Benchmark the runtime of a cold-cache read_parquet."""
        self.run()

    def peakmem_run(self, _data):
        """Benchmark the memory usage of a cold-cache read_parquet."""
        self.run()
