"""Two sample benchmarks to compute runtime and memory usage.

For more information on writing benchmarks:
https://asv.readthedocs.io/en/stable/writing_benchmarks.html."""

import numpy as np
import pandas as pd
import pyarrow as pa
from nested_pandas import NestedDtype


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
            # Sorting is happening somewhere, so we need to order by field name here
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
