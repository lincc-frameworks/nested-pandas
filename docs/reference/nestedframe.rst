=========
NestedFrame
=========
.. currentmodule:: nested_pandas

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   NestedFrame

Nesting
~~~~~~~~~
.. autosummary::
    :toctree: api/
    
    NestedFrame.add_nested
    NestedFrame.from_flat
    NestedFrame.from_lists

Extended Pandas.DataFrame Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note:: 
   The NestedFrame extends the Pandas.DataFrame interface, so all methods
   of Pandas.DataFrame are available. The following methods are extended
   to support NestedFrame functionality. Please reference the Pandas
   documentation for more information.
   https://pandas.pydata.org/docs/reference/frame.html
   
.. autosummary::
    :toctree: api/

    NestedFrame.eval
    NestedFrame.query
    NestedFrame.dropna
    NestedFrame.sort_values
    NestedFrame.reduce

I/O
~~~~~~~~~
.. autosummary::
    :toctree: api/

    NestedFrame.to_parquet
    read_parquet