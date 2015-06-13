# Gather #

## About ##

This program collects some ideas of how to implement gather and scatter
operations for vectorization. It uses SSE2/4, AVX, AVX2 and AVX512.

## Usage ##

Building the program:

    $ make

Creating statistics of of all methods (tries to use different compilers):

    $ ./measure.pl > statistics
    $ less statistics

Measure.pl also creates data files which may be used for Gnuplot.

Creating assembly dumps of the program to get what the compiler generates:

    $ ./disasm.sh

## Dependencies ##

- Compilers: g++ or clang++ or icpc
- Make
- objdump for disasm.sh
- Perl for measure.pl

## License ##

GPL Version 3
