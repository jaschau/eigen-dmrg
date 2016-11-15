CC = clang++
EIGEN_DIR=/usr/local/include/eigen3
SRCDIR=./include
CFLAGS=-I$(SRCDIR) -I$(EIGEN_DIR) 
OPTS=-DNDEBUG -O3 -msse3 -Wall -g -L/usr/local/lib 

.PHONY: clean

img_targetfiles := $(patsubst %.tex,%.svg,$(wildcard doc/img/*.tex))

$(img_targetfiles): %.svg: %.tex
	# $< is the first prerequisite	
	pdflatex -output-directory=$(@D) $<
	# $* is the stem of the target
	pdf2svg $*.pdf $*.svg

doc: eigen_dmrg.doxygen README.md tutorial.md build_instructions.md ising.cc $(img_targetfiles) 
	doxygen $< 

ising: ising.cc
	$(CC) -o ising -larpack -llapack -lblas ising.cc $(OPTS) $(CFLAGS)

clean:
	rm -f ising 
	rm -R doc/html
