=========
NestedFrame
=========
.. currentmodule:: nested_pandas

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   NestedFrame

Helpful Properties
~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree: api/

    NestedFrame.nested_columns
    NestedFrame.base_columns
    NestedFrame.all_columns

Nesting
~~~~~~~~~
.. autosummary::
    :toctree: api/
    
    NestedFrame.join_nested
    NestedFrame.nest_lists
    NestedFrame.from_flat
    NestedFrame.from_lists

Extended Pandas.DataFrame Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note:: 
   The NestedFrame extends the Pandas.DataFrame interface, so all methods
   of Pandas.DataFrame are available. The following methods are a mix of
   newly added methods and extended methods from Pandas DataFrame
   to support NestedFrame functionality. Please reference the Pandas
   documentation for more information.
   https://pandas.pydata.org/docs/reference/frame.html
   
.. autosummary::
    :toctree: api/

    NestedFrame.get_subcolumns
    NestedFrame.eval
    NestedFrame.query
    NestedFrame.dropna
    NestedFrame.sort_values
    NestedFrame.map_rows
    NestedFrame.drop
    NestedFrame.min
    NestedFrame.max
    NestedFrame.describe
    NestedFrame.explode
    NestedFrame.fillna

I/O
~~~~~~~~~
.. autosummary::
    :toctree: api/

    NestedFrame.to_parquet
    NestedFrame.to_pandas
    read_parquet