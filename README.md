Introduction 		{#mainpage}
============

An implementation of the DMRG algorithm using eigen and C++

The density renormalization group (DMRG) provides an algorithm to numerically
determine the ground state of one-dimensional quantum systems through
variational optimization. This project implements the DMRG algorithm using
eigen, ARPACK/BLAS and C++. Special care was taken to make the implementation
as numerically efficient as possible. The project is not as feature-rich as
other more full-fledged DMRG implementations, but the limited functionality
leads to a very short list of dependencies. This makes it possible, in
particular, to compile the project for use with CONDOR which is available in
many universities for distributed computing. 


General usage
-------------

There is a @subpage tutorial and a summary of @subpage build_instructions.

To generate the documentation, just clone the repository and run 

    make doc

Note that this requires pdflatex and pdf2svg for the generation of figures that
were created using tikz. 


