#!/usr/bin/env python

from optparse import OptionParser, OptionValueError
import numpy as np

if __name__ == '__main__':
    usage = 'python ising_memory.py [option]'
    parser = OptionParser(usage=usage)
    parser.add_option('-N', '', type='int', dest='N',
                      help='number of sites')
    parser.add_option('-D', '', type='int', dest='D', help='bond dimension')

    (options, args) = parser.parse_args()
    N = options.N
    D = options.D
    d = 2

    # size of a complex double
    cdbl_size = 8
    # 4 is the bond dimension of the MPO Hamiltonian and to each of the
    # bonds corresponds one transfer matrix of size D*D
    state_memory = N*D*D*d*cdbl_size
    dmrg_memory = N*D*D*4*cdbl_size
    total_memory = state_memory+dmrg_memory

    print '{:.3f} MB'.format(total_memory/(1024.*1000))
