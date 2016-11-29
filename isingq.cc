
#define MKL_Complex16 std::complex<double> 

#include <stdlib.h>
#include <cstdio>
#include <iterator>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

#include <Eigen/Dense>

#include "mps.h"
#include "mpo.h"
#include "dmrg.h"

#include <tclap/CmdLine.h>

typedef std::complex<double> c_double;

#define ISING_DT double

class IsingQMPO : public TransInvMPO<ISING_DT, 2, 2> 
{
public:
  IsingQMPO(std::size_t q, std::size_t N, double h, double J, double kappa);

  std::size_t q;
  Matrix<ISING_DT, 2, 2> pauliX;
  Matrix<ISING_DT, 2, 2> pauliZ;

  Matrix<ISING_DT, 2, 2> opEye;
};

IsingQMPO::IsingQMPO(std::size_t q_, std::size_t N, double h, double J, double kappa) : \
    TransInvMPO<ISING_DT, 2, 2>(N, 3), q(q_)
{
  pauliX << 0, 1,
            1, 0;
  pauliZ << 1, 0,
            0, -1;
  opEye << 1, 0,
           0, 1;
 
  // omegaq = (-1)^q 
  double omegaq = 1;
  for (std::size_t i = 0; i < q; ++i)
    omegaq *= (-1);

  // Matrix<ISING_DT, 2, 2> opOnsite = -h*pauliZ;
  
  ///*
  this->wBulk.setElement(0, 0, opEye, true);
  this->wBulk.setElement(1, 0, pauliX);
  // bulk onsite operator
  this->wBulk.setElement(2, 0, -J*pauliZ - kappa*pauliX);
  this->wBulk.setElement(2, 1, -h*pauliX);
  this->wBulk.setElement(2, 2, opEye, true);

  // on-site operator
  this->wLeft.setElement(0, 0, -J*pauliZ - kappa*pauliX - h*omegaq*pauliX);
  this->wLeft.setElement(0, 1, -h*pauliX);
  this->wLeft.setElement(0, 2, opEye, true);

  this->wRight.setElement(0, 0, opEye, true);
  this->wRight.setElement(1, 0, pauliX);
  // on-site operator
  this->wRight.setElement(2, 0, -J*pauliZ - h*pauliX - kappa*pauliX 
                                + (N+2)*(std::abs(h)+std::abs(J))*opEye);
  //*/

  /*
  this->wBulk.setElement(0, 0, opEye, true);
  this->wBulk.setElement(1, 0, pauliX);
  this->wBulk.setElement(2, 0, pauliX);
  this->wBulk.setElement(3, 0, -J*pauliZ);
  this->wBulk.setElement(3, 1, -0.5*h*pauliX);
  this->wBulk.setElement(3, 2, -0.5*h*pauliX);
  this->wBulk.setElement(3, 3, opEye, true);

  // on-site operator
  this->wLeft.setElement(0, 0, -J*pauliZ - h*omegaq*pauliX);
  this->wLeft.setElement(0, 1, -0.5*h*pauliX);
  this->wLeft.setElement(0, 2, -0.5*h*pauliX);
  this->wLeft.setElement(0, 3, opEye, true);

  this->wRight.setElement(0, 0, opEye, true);
  this->wRight.setElement(1, 0, pauliX);
  this->wRight.setElement(2, 0, pauliX);
  // on-site operator
  this->wRight.setElement(3, 0, -J*pauliZ - h*pauliX 
                                + (N+2)*(std::abs(h)+std::abs(J))*opEye);
  */
}

double convert(double f) 
{
  return f;
}

double convert(c_double f)
{
  return f.real();
}

class OptimizationSession
{
public:
  OptimizationSession(
    std::size_t q_, double h_, double J_, double kappa_, std::size_t Neff, 
    std::size_t D, std::vector<double> tols, bool staggeredMagnetization=false);

  void run();
  std::vector<double> getTolVec(double tol, double tol0);

  std::size_t Neff;
  std::size_t N;
  double h;
  double J;
  double kappa;
  std::vector<double> tols;
  MPS<ISING_DT> mps0;
  IsingQMPO ham;
  bool staggeredMagnetization;

  double qual0;
  double en0;
  double en02;

  double suscep;
  double binderV;
  double suscepStag;
  double binderVStag;
};

OptimizationSession::OptimizationSession(
    std::size_t q_, double h_, double J_, double kappa_, std::size_t Neff_, 
    std::size_t D, std::vector<double> tols_, bool staggeredMagnetization_) :
  Neff(Neff_), N(Neff_-1), h(h_), J(J_), kappa(kappa_), tols(tols_), 
  mps0(std::vector<std::size_t>(N, 2), D), ham(q_, N, h, 1., kappa),
  staggeredMagnetization(staggeredMagnetization_),
  qual0(0), en0(0), en02(0), 
  suscep(0), binderV(0), suscepStag(0), binderVStag(0) 
{
  ISING_DT norm = mps0.norm();
  printf("# initial norm: %.10e\n", convert(norm));
  // std::cout << "# initial norm: " <<  << std::endl;
};

std::vector<double> OptimizationSession::getTolVec(double tol, double tol0)
{
  std::vector<double> tolVec(mps0.N, tol);
  // std::cout << "using tol0 = " << tol0 << std::endl;
  tolVec[0] = tol0;
  tolVec[1] = tol0;
  return tolVec;
};

void OptimizationSession::run()
{  
  TransInvMPO<ISING_DT, 2, 2> ham2 = ham.dot(ham);
  DMRGOptimizer<ISING_DT, 2, 2> dmrg(mps0, ham);
  double en0 = 0;
  for (std::size_t i = 0; i < tols.size(); ++i)
  {
    double tol = tols[i];
    std::vector<double> tolVec = getTolVec(tol, std::min(1e-8, tol));
    en0 = dmrg.fullSweep(tolVec);
  }
  // std::cout << "callCount: " << dmrg.callCount << std::endl;
  double en02 = convert(mps0.expValue(ham2));
  double qual0 = std::sqrt(std::abs(en0*en0 - en02))/std::abs(en0);

  this->en0 = en0;
  this->en02 = en02;
  this->qual0 = qual0; 

  Matrix<ISING_DT, 2, 2> mMag = ham.pauliX;
  double omegaq = 1;
  for (std::size_t i = 0; i < ham.q; ++i)
     omegaq *= -1; 
  // Since the first site does by construction have a magnetization of 
  // cos(2pi/6 q), the total magnetization can be obtained as 
  // sum_{i=1}^N [m_i + cos(2pi/6 q)/N]

  Matrix<ISING_DT,2,2> opMagStag1 = 1./Neff*(
      mMag - omegaq*1./N*Matrix<ISING_DT,2,2>::Identity());
  Matrix<ISING_DT,2,2> opMagStag2 = 1./Neff*(
      -mMag - omegaq*1./N*Matrix<ISING_DT,2,2>::Identity());
  Matrix<ISING_DT,2,2> opMag = 1./Neff*(
      mMag + omegaq*1./N*Matrix<ISING_DT,2,2>::Identity());

  TransInvSingleSiteMPO<ISING_DT, 2, 2> mpoMagStag(
      mps0.N, 
      opMagStag1,
      opMagStag2);
  TransInvSingleSiteMPO<ISING_DT, 2, 2> mpoMag(
      mps0.N, 
      opMag);
  TransInvMPO<ISING_DT, 2, 2> mpoMag2 = mpoMag.dot(mpoMag);
  TransInvMPO<ISING_DT, 2, 2> mpoMag3 = mpoMag2.dot(mpoMag);
  TransInvMPO<ISING_DT, 2, 2> mpoMag4 = mpoMag2.dot(mpoMag2);
  TransInvMPO<ISING_DT, 2, 2> mpoMagStag2 = mpoMagStag.dot(mpoMagStag);
  TransInvMPO<ISING_DT, 2, 2> mpoMagStag3 = mpoMagStag2.dot(mpoMagStag);
  TransInvMPO<ISING_DT, 2, 2> mpoMagStag4 = mpoMagStag2.dot(mpoMagStag2);

  double mag = convert(mps0.expValue(mpoMag));
  double mag2 = convert(mps0.expValue(mpoMag2));
  double mag3 = convert(mps0.expValue(mpoMag3));
  double mag4 = convert(mps0.expValue(mpoMag4));
  double magStag = convert(mps0.expValue(mpoMagStag));
  double magStag2 = convert(mps0.expValue(mpoMagStag2));
  double magStag3 = convert(mps0.expValue(mpoMagStag3));
  double magStag4 = convert(mps0.expValue(mpoMagStag4));

  this->binderV = 1. - mag4/(3.*mag2*mag2);
  this->binderVStag = 1. - magStag4/(3.*magStag2*magStag2);
  this->suscep = Neff*Neff*(mag2 - mag*mag);
  this->suscepStag = Neff*Neff*(magStag2 - magStag*magStag);

}


template <typename T>
void parseCsv(const std::string& s_, std::vector<T>& vals)
{
  std::string s = s_;
  std::replace(s.begin(), s.end(), ',', ' ');
  std::stringstream ss(s);
  std::copy(std::istream_iterator<T>(ss), std::istream_iterator<T>(), 
            std::back_inserter(vals));
}

template <typename T>
std::string toStr(const std::vector<T>& v)
{
  std::ostringstream ss;
  for (std::size_t i = 0; i < v.size(); ++i)
  {
    ss << v[i]; 
    if (i < v.size()-1)
      ss << ",";
  }
  return ss.str();
}



int maina(int argc, char** argv) 
{
  typedef Matrix<ISING_DT, Dynamic, Dynamic> Mat;

  std::size_t N = 8;
  std::size_t D = 32;
  std::size_t q = 0;
  double h = 1.;
  double J = 0.;
  IsingQMPO hamIsing = IsingQMPO(q, N, h, J, 0.);

  std::vector<std::size_t> dims(N, 2);
  MPS<ISING_DT> mps0(dims, D);
  std::size_t ntols = 3; 
  double tols[] = {1e-6, 1e-8, 1e-10};
  DMRGOptimizer<ISING_DT, 2, 2> dmrg0(mps0, hamIsing);
  double en0 = 0;
  for (std::size_t i = 0; i < ntols; ++i)
  {
    en0 = dmrg0.fullSweep(tols[i]);
  }
  printf("en = %e\n", en0);
  
  return 0;
}


int main(int argc, char** argv)
{
  // setenv("VECLIB_MAXIMUM_THREADS", "1", true);
  std::vector<int> valsD;
  std::vector<double> valsTol;
  std::size_t N;
  std::size_t q;
  double h, J;
  J = 1.;
  double kappa;
  bool staggeredMagnetization;
  try 
  {  
    char cmdDelim = ' ';
    TCLAP::CmdLine cmd("Command description message", cmdDelim, "1.0");

    TCLAP::ValueArg<std::size_t> argN("N", "size", "chain length", false, 32, 
        "int");
    TCLAP::ValueArg<std::string> argD("D", "bond", "bond dimension", false, 
        "32", "csv list of int");
    TCLAP::ValueArg<double> argh("l", "lambda", 
        "This parameter is called h in the Hamiltonian, but since -h is "
        "reserved for help, we call it lambda here.", false, 0.1, "double");
    TCLAP::ValueArg<std::size_t> argq("q", "q", "the Z6 parity sector to use", 
        false, 0, "unsigned int");
    TCLAP::ValueArg<std::string> argTols("t", "tols", "tolerances", false, 
        "1e-4,1e-6,1e-8,1e-10", "csv list of double");
    TCLAP::SwitchArg argStaggeredMagnetization("s", "staggered_magnetization", 
        "toggles staggering of magnetization", false);  
    TCLAP::ValueArg<double> argkappa("k", "kappa", "kappa", false, 0.,
        "double");

    cmd.add(argN);
    cmd.add(argD);
    cmd.add(argh);
    cmd.add(argTols);
    cmd.add(argq);
    cmd.add(argkappa);
    cmd.add(argStaggeredMagnetization);

    // Parse the argv array.
    cmd.parse(argc, argv);
    N = argN.getValue();
    h = argh.getValue(); 
    q = argq.getValue() % 2;
    kappa = argkappa.getValue();
    staggeredMagnetization = argStaggeredMagnetization.getValue();
    parseCsv<int>(argD.getValue(), valsD);
    parseCsv<double>(argTols.getValue(), valsTol); 
  } 
  catch (TCLAP::ArgException &e)  // catch any exceptions
  { 
    std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl; 
    return -1;
  }
 

  printf("# q=%lu, N=%lu, D=%s, h=%.6f, tols=%s\n", 
         q, N, toStr(valsD).c_str(), h, toStr(valsTol).c_str());
  printf("# order is h, kappa, N, D, q, en0, qual0");
  printf(", suscep, binderV, suscepStag, binderVStag\n");
  for (std::size_t i = 0; i < valsD.size(); ++i) 
  {
    std::size_t D = valsD[i];
    
    OptimizationSession optSession(q, h, J, kappa, N, D, valsTol, 
        staggeredMagnetization);
    optSession.run();

    printf("%.6f, %.6f, %lu, %lu, %lu, %.10e, %.10e ", h, kappa, N, D, q, 
           optSession.en0, optSession.qual0);
    printf(", %.10e, %.10e, %.10e, %.10e\n", 
           optSession.suscep, optSession.binderV, optSession.suscepStag, 
           optSession.binderVStag);
  }


  return 0;
}
