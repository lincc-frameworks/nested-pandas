Installation
============

nested-pandas is available to install with pip, using the "nested-pandas" package name:

.. code-block:: bash

    % pip install nested-pandas


This will grab the latest release version of nested-pandas from pip.

Installation from Source
---------------------

In some cases, installation via pip may not be sufficient. In particular, if you're looking to grab the latest
development version of nested-pandas, you should instead build 'nested-pandas' from source. The following process downloads the 
'nested-pandas' source code and installs it and any needed dependencies in a fresh conda environment. 

.. code-block:: bash

    conda create -n nested_pandas_env python=3.11
    conda activate nested_pandas_env

    git clone https://github.com/lincc-frameworks/nested-pandas.git
    cd nested-pandas
    pip install .
    pip install .[dev]  # it may be necessary to use `pip install .'[dev]'` (with single quotes) depending on your machine.

The ``pip install .[dev]`` command is optional, and installs dependencies needed to run the unit tests and build
the documentation. The latest source version of nested-pandas may be less stable than a release, and so we recommend 
running the unit test suite to verify that your local install is performing as expected.

.. code-block:: bash

    pip install pytest
    pytest