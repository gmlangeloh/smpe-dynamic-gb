# smpe-dynamic-gb

This repository includes some experiments and data on Dynamic Gröbner Basis algorithms.
The analysis of the data is made in the analysis.Rmd file and is reported in
analysis.html.

In order to generate the data of the data subdirectory (not recommended, it takes days),
it is necessary to run

`sage test.sage`

preferably with the output redirected to a file. We used Sage 8.8 with Python 3.7.3.
Currently, in order to support Python 3, Sage needs to be compiled from source.

The Sage source code can be obtained at

http://www.sagemath.org/download.html

and instructions for compilation with support for Python 3 are available at

https://wiki.sagemath.org/Python3-compatible%20code
