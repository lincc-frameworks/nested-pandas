"""Two sample benchmarks to compute runtime and memory usage.

For more information on writing benchmarks:
https://asv.readthedocs.io/en/stable/writing_benchmarks.html."""

import numpy as np
import pandas as pd
import pyarrow as pa
from nested_pandas import NestedDtype, NestedFrame, datasets


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
            dtype=NestedDtype.from_fields({"band": pa.string(), "flux": pa.float64(), "time": pa.float64()}),
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
        dtype = NestedDtype.from_fields({"band": pa.string(), "flux": pa.float64(), "time": pa.float64()})
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
    """Benchmark the NestedFrame.add_nested function"""

    n_base = 100
    layer_size = 1000
    base_nf = NestedFrame
    layer_nf = NestedFrame

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
        self.base_nf.add_nested(self.layer_nf, "nested")

    def time_run(self):
        """Benchmark the runtime of adding a nested layer"""
        self.run()

    def peakmem_run(self):
        """Benchmark the memory usage of adding a nested layer"""
        self.run()


class NestedFrameReduce:
    """Benchmark the NestedFrame.reduce function"""

    n_base = 100
    n_nested = 1000
    nf = NestedFrame

    def setup(self):
        """Set up the benchmark environment"""
        self.nf = datasets.generate_data(self.n_base, self.n_nested)

    def run(self):
        """Run the benchmark."""
        self.nf.reduce(np.mean, "nested.flux")

    def time_run(self):
        """Benchmark the runtime of applying the reduce function"""
        self.run()

    def peakmem_run(self):
        """Benchmark the memory usage of applying the reduce function"""
        self.run()


class NestedFrameQuery:
    """Benchmark the NestedFrame.query function"""

    n_base = 100
    n_nested = 1000
    nf = NestedFrame

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
