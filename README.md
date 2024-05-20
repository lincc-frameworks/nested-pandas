# nested-pandas

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/lincc-frameworks/nested-pandas/smoke-test.yml)](https://github.com/lincc-frameworks/nested-pandas/actions/workflows/smoke-test.yml)
[![codecov](https://codecov.io/gh/lincc-frameworks/nested-pandas/branch/main/graph/badge.svg)](https://codecov.io/gh/lincc-frameworks/nested-pandas)
[![Read the Docs](https://img.shields.io/readthedocs/nested-pandas)](https://nested-pandas.readthedocs.io/)
[![benchmarks](https://img.shields.io/github/actions/workflow/status/lincc-frameworks/nested-pandas/asv-main.yml?label=benchmarks)](https://lincc-frameworks.github.io/nested-pandas/)

An extension of pandas for efficient representation of nested
associated datasets.

Nested-Pandas extends the [pandas](https://pandas.pydata.org/) package with 
tooling and support for nested dataframes packed into values of top-level 
dataframe columns. [Pyarrow](https://arrow.apache.org/docs/python/index.html) 
is used internally to aid in scalability and performance.

![image](./nestedframe.png)

Nested-Pandas is motivated by time-domain astronomy use cases, where we see
typically two levels of information, information about astronomical objects and
then an associated set of `N` measurements of those objects. Nested-Pandas offers
a performant and memory-efficient package for working with these types of datasets. 

Core advantages being:
* hierarchical column access
* efficient packing of nested information into inputs to custom user functions
* avoiding costly groupby operations



This is a LINCC Frameworks project - find more information about LINCC Frameworks [here](https://lsstdiscoveryalliance.org/programs/lincc-frameworks/).



## Acknowledgements

This project is supported by Schmidt Sciences.
