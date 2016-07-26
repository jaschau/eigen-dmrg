#ifndef __SVD_H
#define __SVD_H

#include <Eigen/Dense>
#include <cstdlib>

typedef  std::complex<double> c_double;

extern "C"  {
  void dgesvd_ (char* jobu, char* jobv, int* m, int* n,
     double* a, int* lda, double* s, double *u, int* ldu,
      double* vt, int* ldvt, double* work, int* lwork, int* info);

  void zgesvd_ (char* jobu, char* jobv, int* m, int* n,
      c_double* a, int* lda, double* s, c_double* u, int* ldu,
      c_double* vt, int* ldvt, c_double* work, int* lwork, double* rwork,
      int* info);
}

/**
 * Computes the singular value decomposition of a double-valued matrix A using BLAS 
 * and returns U, S, Vt with A = U * S.asDiagonal() * Vt
 */
template<typename Derived>
int svd(const Eigen::MatrixBase<Derived>& A, 
    Eigen::MatrixXd& U, Eigen::VectorXd& S, Eigen::MatrixXd& Vt) {
  int m= A.rows();
  int n= A.cols();
  int min= std::min(m,n);

  // make a copy of A because the fortran routine destroys the data
  Eigen::MatrixXd a= A;

  // adjust the sizes of the output matrices
  U.resize(m, min);
  S.resize(min);
  Vt.resize(min, n);

  int lwork, info;
  lwork= -1; //query workspace

  char job[2]= "S";

  double work_s;

  dgesvd_(job, job, &m, &n, a.data(), &m, S.data(), U.data(), &m, Vt.data(),
      &min, &work_s, &lwork, &info );

  // allocate workspace
  lwork= int(work_s);
  double* work= new double[lwork];
  dgesvd_(job, job, &m, &n, a.data(), &m, S.data(), U.data(), &m, Vt.data(),
      &min, work, &lwork, &info );
  delete[] work;

  if (info < 0 ) 
    std::cerr << "the  " << -info << "-th argument had an illegal value";
  else if (info > 0)
  {
    std::cerr << "svd: algorithm did not converge" << std::endl;
    std::exit(1);
  }
  return info;
}


/**
 * Computes the singular value decomposition of a complex double-valued matrix 
 * A using BLAS and returns U, S, Vt with A = U * S.asDiagonal() * Vt
 */
template<typename Derived>
int svd(const Eigen::MatrixBase<Derived>& A, 
    Eigen::MatrixXcd& U, Eigen::VectorXd& S, Eigen::MatrixXcd& Vt) {
  int m= A.rows();
  int n= A.cols();
  int min= std::min(m,n);

  // make a copy of A because the fortran routine destroys the data
  Eigen::MatrixXcd a= A;

  // adjust the sizes of the output matrices
  U.resize(m, min);
  S.resize(min);
  Vt.resize(min, n);

  int lwork, info;
  lwork= -1; //query workspace

  char job[2]= "S";

  c_double work_s;
  double* rwork= new double[5*min];

  zgesvd_(job, job, &m, &n, a.data(), &m, S.data(), U.data(), &m, Vt.data(),
      &min, &work_s, &lwork, rwork, &info );

  // allocate workspace
  lwork= int(work_s.real());
  c_double* work= new c_double[lwork];
  zgesvd_(job, job, &m, &n, a.data(), &m, S.data(), U.data(), &m, Vt.data(),
      &min, work, &lwork, rwork, &info );
  delete[] rwork;
  delete[] work;

  if (info < 0 ) 
    std::cerr << "the  " << -info << "-th argument had an illegal value";
  else if (info > 0)
  {
    std::cerr << "svd: algorithm did not converge" << std::endl;
    std::exit(1);
  }
  return info;
}

#endif /* __SVD_H */
