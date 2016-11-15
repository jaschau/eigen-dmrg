Build instructions  {#build_instructions}
------------------

### Dependencies ###

This project depends on [eigen](http://eigen.tuxfamily.org), BLAS (any BLAS
library should do) and
[arpack](https://forge.scilab.org/index.php/p/arpack-ng). For the examples, I
also use [TCLAP](http://tclap.sourceforge.net) as a command-line parser, but
the library itself does not depend on it.

Assuming that the header files for blas, arpack and lapack are on your PATH and 
the compiled libraries are on your LD_LIBRARY_PATH, a minimal build file could 
look like this:

    CC = clang++
    EIGEN_DIR=/usr/local/include/eigen3
    TCLAP_DIR=./ext_libs/tclap-1.2.1/include 
    SRCDIR=./include
    CFLAGS=-I$(SRCDIR) -I$(EIGEN_DIR) -I$(TCLAP_DIR)
    OPTS=-DNDEBUG -O3 -msse3 -Wall -g 

    isingq: isingq.cc
        $(CC) -o isingq -larpack -llapack -lblas isingq.cc $(OPTS) $(CFLAGS)

Note that the usage of the options "-DNDEBUG -O3 -msse3" are crucial in order
to achieve the best performance.  
 

### Usage with CONDOR ### 

In order to use the advanced features of CONDOR such as checkpointing in the
standard universe, the compiled code has to be relinked using the program
condor_compile. Using condor_compile brings about certain limitations, in
particular, neither the program code nor the used libraries may use threads.
This means that besides eigen-dmrg, one also has to compile ARPACK and BLAS
with threading deactivated.

I will explain a possible installation using 
[openblas](http://www.openblas.net)
and [arpack-ng](https://github.com/opencollab/arpack-ng).

-   _openblas_ 
    
    In order to deactivate threading, simply use the option USE_THREAD=0 with make.
    Since checkpointing in CONDOR may lead to code being executed on different
    architectures, I found it necessary to specify a processor architecture with a
    minimal feature set that is supported by all computers in the CONDOR cluster
    using the option TARGET=XXX, where XXX can be one of several architectures
    explained in the file TargetList.txt. 
    Example build:

        make USE_THREAD=0 TARGET=OPTERON
        make install PREFIX=YOURPREFIX

    YOURPREFIX has to be replaced with the desired output directory for the build.

-   _arpack-ng_ 

    arpack-ng only needs to be told where to find the openblas installation, using

        ./configure --with-blas=OPENBLAS_PREFIX --with-lapack=YOURPREFIX \
        --prefix=YOURPREFIX

With openblas and arpack-ng properly configured, your program can be linked for
usage with condor using condor_compile: 

    condor_compile $(CC) -static PROGRAM.o -o PROGRAM larpack -lopenblas \
    -lgfortran -LYOURPREFIX/lib

 
