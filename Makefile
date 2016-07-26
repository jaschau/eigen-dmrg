CC = clang++
EIGEN_DIR=/usr/local/include/eigen3
TCLAP_DIR=./ext_libs/tclap-1.2.1/include 
SRCDIR=./include
CFLAGS=-I$(SRCDIR) -I$(EIGEN_DIR) -I$(TCLAP_DIR)
OPTS=-DNDEBUG -O3 -msse3 -Wall -g 


isingq: isingq.cc
	$(CC) -o isingq -larpack -llapack -lblas isingq.cc $(OPTS) $(CFLAGS)

#paraq: paraq.cc
#	$(CC) -o paraq -larpack -llapack -lblas paraq.cc $(OPTS) $(CFLAGS)

.PHONY: clean

clean:
	rm -f isingq 
	rm -R doc/*
