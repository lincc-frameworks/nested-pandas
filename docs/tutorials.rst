Tutorials
========================================================================================

.. toctree::

    Loading Data into Nested-Pandas <tutorials/data_loading_notebook>
    Fine Data Manipulation with Nested-Pandas <tutorials/data_manipulation>
    Lower-level interfaces <tutorials/low_level.ipynb>
    Using Nested-Pandas with Astronomical Spectra <pre_executed/nested_spectra.ipynb>
    Using GroupBy with Nested-Pandas <tutorials/groupby_doc.ipynb>

Working with Nested Data
________________________

Here are simple ways to create and manipulate nested structures.

**1. Creating Nested Data (The Pack Way)**
This creates a normal flat table and groups it into nested rows.

.. code-block:: python

	import pandas as pd
	import nested_pandas as npd

	#Example:For two stars each with two measurements of brightness
	data = {
		"star_id": ["Star_A","Star_A","Star_B",Star_B"],
		"brightness": [15.3,15.4,12.5,12.6]
	}
	df = pd.DataFrame(data)

	#In order to get one row per star, with all brightness values in a nested list
	nested_stars = df.groupby("star_id").nested.pack(name="light_curve)
	print(nested_stars)

**2. Advanced Manipulation with .eval()**
In order to perform fast calculations on nested columns using the ".eval()" method. This is much faster than manually using loops.

.. code-block:: python

	#Example:For adding a calibration constant to a nested magnitude column
	#This creates 'calibrated_mag' by adding 25.5 to 'mag' inside the 'lc' nested column
	nf2 = nf.eval("lc.calibrated_mag = lc.mag + 25.5")
	
	#Example: Combining flat columns with nested aggregations
	#This adds a flat value 'a' to the median of nested column 'c'
	nf2 = nf.eval("result = a + packed.c.median()")

