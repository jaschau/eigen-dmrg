#include <stdlib.h>
#include <cstdio>
#include <iterator>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

#include "mps.h"
#include "mpo.h"
#include "dmrg.h"


class IsingMPO : public TransInvMPO<double, 2, 2> 
{
public:
  IsingMPO(std::size_t N, double J);

  Matrix<double, 2, 2> pauliX;
  Matrix<double, 2, 2> pauliZ;
  Matrix<double, 2, 2> opEye;
};

IsingMPO::IsingMPO(std::size_t N, double J) : TransInvMPO<double, 2, 2>(N, 3) 
{
  pauliX << 0, 1,
        1, 0;
  pauliZ << 1, 0,
        0, -1;
  opEye << 1, 0,
       0, 1;
  double h = 1;

  this->wBulk.setElement(0, 0, opEye, true);
  this->wBulk.setElement(1, 0, pauliZ);
  this->wBulk.setElement(2, 0, -h*pauliX);
  this->wBulk.setElement(2, 1, -J*pauliZ);
  this->wBulk.setElement(2, 2, opEye, true);

  this->wLeft.setElement(0, 0, -h*pauliX);
  this->wLeft.setElement(0, 1, -J*pauliZ);
  this->wLeft.setElement(0, 2, opEye, true);

  this->wRight.setElement(0, 0, opEye, true);
  this->wRight.setElement(1, 0, pauliZ);
  this->wRight.setElement(2, 0, -h*pauliX);
}

int main(int argc, char** argv) {
  // Internal bond dimension of the MPS
  std::size_t D = 32;
  // Size of the chain
  std::size_t N = 16;
  double J = 0.;
  // Create a MPS with N sites of physical dimension d=2 and internal bond dimension D
  MPS<double> mps(std::vector<std::size_t>(N, 2), D);
  // MPO representation of the Hamiltonian
  IsingMPO ham(N, J);
  // Square of the Hamiltonian as a MPO
  TransInvMPO<double, 2, 2> ham2 = ham.dot(ham);
  // Class performing the variational optimization of mps for the ground state search of ham 
  DMRGOptimizer<double, 2, 2> dmrg(mps, ham);
  // specifiy the numerical precision values to use in each sweep 
  double tols[] = {1e-4,1e-6,1e-8,1e-10};
  std::size_t ntols = 4;
  double en = 0;
  // perform as many sweeps as there are elements in tol
  for (std::size_t i = 0; i < ntols; ++i)
  {
    std::cout << "Performing run " << i << " at precision " << tols[i] 
      << std::endl;
    double tol = tols[i];
    en = dmrg.fullSweep(tol);
  }
  double en2 = mps.expValue(ham2);
  // get a variational estimate how close we are to a gound ground state
  // approximation
  double qual = std::sqrt(std::abs(en*en - en2))/std::abs(en);
  std::cout << "Energy is " << en << std::endl;
  // outputs ~ 1e-8 at my system
  std::cout << "Qual is " << qual << std::endl;

  // evaluate order parameter
  TransInvSingleSiteMPO<double, 2, 2> mpoMag(mps.N, 
    1./mps.N*ham.pauliX);
  double mag = mps.expValue(mpoMag);
  
  std::cout << "mag: " << mag << std::endl;

  return 0;
}
