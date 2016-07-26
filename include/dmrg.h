#ifndef __DMRG_H
#define __DMRG_H

#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <complex>
#include <cstdlib>
#include "mps.h"
#include "transfer_operator_buffer.h"

typedef  std::complex<double> c_double;

const c_double I= c_double(0,1);

#ifndef EIGEN_USE_BLAS
// BLAS routines
extern "C" {
  void dgemm_(char* transA, char* transB, int *m, int* n, int* k, 
              double* alpha, const double* A, int *lda, const double* B, 
              int *ldb, double* beta, double* C, int *ldc);

  void zgemm_(char* transA, char* transB, int *m, int* n, int* k,
              c_double* alpha, const c_double* A, int *lda, const c_double* B, 
              int *ldb, c_double* beta, c_double* C, int *ldc);
}
#endif

void gemm(char* transA, char* transB, int *m, int* n, int* k,
          double* alpha, const double* A, int *lda, const double* B, int *ldb, 
          double* beta, double* C, int *ldc);

void gemm(char* transA, char* transB, int *m, int* n, int* k,
          c_double* alpha, c_double* A, int *lda, c_double* B, int *ldb, 
          c_double* beta, c_double* C, int *ldc);

// ARPACK routines
extern "C" {
  //http://forge.scilab.org/index.php/p/arpack-ng/source/tree/master/SRC/dnaupd.f
  void dsaupd_(int *ido, char *bmat, int *n, char *which, int *nev, 
               double *tol, double *resid, int *ncv, double *v, int *ldv, 
               int *iparam, int *ipntr, double *workd, double *workl, 
               int *lworkl, int *info);


  void dseupd_(int *rvec, char *All, int *select, double *d, double *z, 
               int *ldz, double *sigma, char *bmat, int *n, char *which, 
               int *nev, double *tol, double *resid, int *ncv, double *v,
               int *ldv, int *iparam, int *ipntr, double *workd,
               double *workl, int *lworkl, int *ierr);

  //http://forge.scilab.org/index.php/p/arpack-ng/source/tree/master/SRC/znaupd.f
  void znaupd_(int *ido, char *bmat, int *n, char *which, int *nev, 
               double *tol, c_double *resid, int *ncv, c_double *v, int *ldv,
               int *iparam, int *ipntr, c_double *workd,
               c_double *workl, int *lworkl, double *rwork, int *info);

  //http://forge.scilab.org/index.php/p/arpack-ng/source/tree/master/SRC/zneupd.f
  void zneupd_(int *rvec, char *howmny, int *select, c_double *d, c_double *z, 
               int *ldz, c_double *sigma, c_double *workev, char *bmat,
               int *n, char *which, int *nev, double *tol, c_double *resid, 
               int *ncv, c_double *v, int *ldv, int *iparam, int *ipntr, 
               c_double *workd, c_double *workl, int *lworkl, double *rwork,
               int* info);
}

void gemm(char* transA, char* transB, int *m, int* n, int* k,
          double* alpha, const double* A, int *lda, const double* B, int *ldb, 
          double* beta, double* C, int *ldc)
{
  dgemm_(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void gemm(char* transA, char* transB, int *m, int* n, int* k,
          c_double* alpha, const c_double* A, int *lda, const c_double* B, 
          int *ldb,  c_double* beta, c_double* C, int *ldc)
{
  zgemm_(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}


template<typename T>
class MPSProjector : public TransferOperatorBuffer< T, MPSProjector<T> >
{
  typedef Matrix<T, Dynamic, Dynamic> Mat;
  typedef Matrix<T, Dynamic, 1> Vec;

public:
  MPSProjector(const MPS<T>& mps, const MPS<T>& orthogonalMps);

  Vec getProjectionVector();
  Mat moveLeft();
  Mat moveRight();

  const MPS<T>& mps;
  const MPS<T>& orthogonalMps;
};

/**
 * A class which variationally determines the matrix product state corresponding to the
 * smallest eigenvalue of a matrix product operator by sweeping through the %MPS and 
 * successively optimizing the individual tensors.
 */
template<typename T, int _Rows, int _Cols>
class DMRGOptimizer : public TransferOperatorBuffer< T, DMRGOptimizer<T,_Rows,_Cols> >
{
  typedef Matrix<T, Dynamic, Dynamic> Mat;
  typedef Matrix<T, Dynamic, 1> Vec;

public:
  DMRGOptimizer(MPS<T>& mps, const MPO<T, _Rows, _Cols>& mpo);

  void addProjector(MPSProjector<T>& proj);
  void updateProjectionMatrix();
  void applyQuadraticForm(T* in, T* out, Vec& tmp);
  void applyQuadraticForm(T* in, T* out); 
  void initMps();

  /** 
   * \link MPS::rightNormalize(std::size_t) Right-normalizes\endlink the %MPS
   * tensor at the \link getPosition() current position\endlink. Returns the
   * right-contraction of that tensor with the buffered transfer operator to
   * the \link getTopRight() right of the current position\endlink.
   */
  Mat moveLeft();

  /** 
   * \link MPS::leftNormalize(std::size_t) Left-normalizes\endlink the %MPS
   * tensor at the \link getPosition() current position\endlink. Returns the
   * left-contraction of that tensor with the buffered transfer operator to the
   * \link getTopLeft() left of the current position\endlink.
   */
  Mat moveRight();


  double fullSweep(double tol=0., int ncv=0) 
  {
    return fullSweep(std::vector<double>(mps.N, tol), ncv);
  }
  double fullSweep(const std::vector<double>& tols, int ncv=0);
  double optimizeEnergy(double tol=0., int ncv=0);

  MPS<T>& mps;
  const MPO<T, _Rows, _Cols>& mpo;
  std::size_t callCount;
private:
  Mat projectionMatrix;
  std::vector< MPSProjector<T>* > projectors;
};

template <int _Rows, int _Cols>
double _optimizeEnergy(DMRGOptimizer<double, _Rows, _Cols>& dmrg, 
                       double tol=0., int ncv=0);

template <int _Rows, int _Cols>
double _optimizeEnergy(DMRGOptimizer<c_double, _Rows, _Cols>& dmrg, 
                       double tol=0., int ncv=0);


template<typename T>
MPSProjector<T>::MPSProjector(const MPS<T>& mps_, const MPS<T>& orthogonalMps_) :
  TransferOperatorBuffer< T, MPSProjector<T> >(mps_.N, -1), 
  mps(mps_), orthogonalMps(orthogonalMps_)
{
}

template<typename T>
Matrix<T,Dynamic,Dynamic> MPSProjector<T>::moveLeft()
{ 
  std::size_t optPos = this->getPosition();
  std::size_t d, Dl, Dr;
  d = mps.getPhysDim(optPos);
  mps.getBondDimensions(optPos, Dl, Dr);

  /*
  Mat topRight = Mat::Ones(1,1);
  for (std::size_t i = mps.N-1; i > optPos; --i)
  {
    std::size_t d1, Dl1, Dr1;
    d1 = mps.getPhysDim(i);
    mps.getBondDimensions(i, Dl1, Dr1);
    topRight = mps.rightContractedTop(d1, Dl1, Dr1, orthogonalMps.tensorAt(i),
      mps.tensorAt(i), topRight);
  }
  std::cout << "moveLeft, norm: " << (topRight-this->getTopRight()).norm() << std::endl;
  */ 

  return mps.rightContractedTop(d, Dl, Dr, orthogonalMps.tensorAt(optPos),
      mps.tensorAt(optPos), this->getTopRight());
}

template<typename T>
Matrix<T,Dynamic,Dynamic> MPSProjector<T>::moveRight()
{ 
  std::size_t optPos = this->getPosition();
  std::size_t d, Dl, Dr;
  d = mps.getPhysDim(optPos);
  mps.getBondDimensions(optPos, Dl, Dr);

  /* 
  Mat topLeft = Mat::Ones(1,1);
  for (std::size_t i = 0; i < optPos; ++i)
  {
    std::size_t d1, Dl1, Dr1;
    d1 = mps.getPhysDim(i);
    mps.getBondDimensions(i, Dl1, Dr1);
    topLeft = mps.leftContractedTop(d1, Dl1, Dr1, orthogonalMps.tensorAt(i),
      mps.tensorAt(i), topLeft);
  }
  // std::cout << "topLeft: " << topLeft << std::endl;
  // std::cout << "this->getTopLeft(): " << this->getTopLeft() << std::endl;
  std::cout << "moveRight, norm: " << (topLeft-this->getTopLeft()).norm() << std::endl;
  */ 
  return mps.leftContractedTop(d, Dl, Dr, orthogonalMps.tensorAt(optPos),
      mps.tensorAt(optPos), this->getTopLeft());
}

template<typename T>
Matrix<T,Dynamic,1> MPSProjector<T>::getProjectionVector()
{
  std::size_t optPos = this->getPosition();
  std::size_t d, Dl, Dr;
  d = orthogonalMps.getPhysDim(optPos);
  orthogonalMps.getBondDimensions(optPos, Dl, Dr);
  const Mat& mL = this->getTopLeft();
  const Mat& mR = this->getTopRight();
  const Vec& mA = orthogonalMps.tensorAt(optPos);

  Vec res(d*Dl*Dr);
  assert(mL.size() == Dl*Dl);
  assert(mR.size() == Dr*Dr);
  assert(mA.size() == d*Dl*Dr);
  Map<Mat, Aligned>(res.data(), Dl, d*Dr).noalias()
    = mL.transpose() * Map<const Mat, Aligned>(mA.data(), Dl, d*Dr).conjugate();
  Map<Mat, Aligned>(res.data(), Dl*d, Dr) 
    = Map<const Mat, Aligned>(res.data(), Dl*d, Dr)*mR.transpose(); 
  res.normalize();  
  return res.conjugate();
}

template<typename T, int _Rows, int _Cols>
DMRGOptimizer<T,_Rows,_Cols>::DMRGOptimizer(
    MPS<T>& mps_, const MPO<T, _Rows, _Cols>& mpo_):
   TransferOperatorBuffer< T, DMRGOptimizer<T,_Rows,_Cols> >(mps_.N, -1),
   mps(mps_), mpo(mpo_), callCount(0)
{
  initMps();
}

template<typename T, int _Rows, int _Cols>
void DMRGOptimizer<T,_Rows,_Cols>::initMps()
{
  for (std::size_t i = mps.N-1; i > 0; --i) 
  {
    this->moveBuffer();
  }

  Mat top = mps.rightContractedTop(0);
  mps.tensorAt(0) *= 1./std::sqrt(std::abs(top(0,0))); 
}

template<typename T, int _Rows, int _Cols>
void DMRGOptimizer<T,_Rows,_Cols>::addProjector(MPSProjector<T>& projector)
{
  projectors.push_back(&projector);
  for (std::size_t i = mps.N-1; i > 0; --i) 
  {
    projector.moveBuffer();
  }
  updateProjectionMatrix();
}

template<typename T, int _Rows, int _Cols>
void DMRGOptimizer<T,_Rows,_Cols>::updateProjectionMatrix()
{
  if (projectors.size() == 1)
  {
    // std::size_t optPos = projectors[0]->getPosition();
    projectionMatrix = projectors[0]->getProjectionVector();

    /*
    std::size_t d = mps.getPhysDim(optPos);
    std::size_t Dl, Dr;
    mps.getBondDimensions(optPos, Dl, Dr);

    std::cout << "Updating projection matrix at site " << optPos << std::endl;
    printf("d=%lu, Dl=%lu, Dr=%lu\n", d, Dl, Dr);
    assert(projectionMatrix.size() == mps.tensorAt(optPos).size());
    c_double overlap = (projectionMatrix.transpose()*mps.tensorAt(optPos))(0,0);
    std::cout << "overlap: " << overlap << std::endl;

    Mat topLeftTotal = Mat::Ones(1,1);
    Mat topLeft = Mat::Ones(1,1);
    for (std::size_t i = 0; i < mps.N; ++i)
    {
      std::size_t d1, Dl1, Dr1;
      d1 = mps.getPhysDim(i);
      mps.getBondDimensions(i, Dl1, Dr1);
      topLeftTotal = mps.leftContractedTop(d1, Dl1, Dr1, 
        projectors[0]->orthogonalMps.tensorAt(i), mps.tensorAt(i), topLeftTotal);
      if (i == optPos-1)
      {
        topLeft = topLeftTotal;
      }
    }
    c_double overlap1 = topLeftTotal(0,0);
    Mat topRight = Mat::Ones(1,1);
    for (std::size_t i = mps.N-1; i > optPos; --i)
    {
      std::size_t d1, Dl1, Dr1;
      d1 = mps.getPhysDim(i);
      mps.getBondDimensions(i, Dl1, Dr1);
      topRight = mps.rightContractedTop(d1, Dl1, Dr1, projectors[0]->orthogonalMps.tensorAt(i),
        mps.tensorAt(i), topRight);
    }

    std::cout << "topRight, norm: " << (topRight-projectors[0]->getTopRight()).norm() << std::endl;
    std::cout << "topLeft norm: " << (topLeft - projectors[0]->getTopLeft()).norm() << std::endl;
    std::cout << "overlap1: " << overlap1 << std::endl;
    */
  }
  else
  {
    // TODO: implement me
    //projectionMatrix = Mat::Zero();
    //colPivHouseholderQr 
  }
}

template<typename T, int _Rows, int _Cols>
Matrix<T,Dynamic,Dynamic> DMRGOptimizer<T,_Rows,_Cols>::moveLeft()
{
  std::size_t optPos = this->getPosition();
  mps.rightNormalize(optPos);
  if (projectors.size() > 0)
  {
    for (std::size_t i = 0; i < projectors.size(); ++i)
      projectors[i]->moveBuffer();
    updateProjectionMatrix();
  }
  return mps.rightContractedTop(optPos, this->getTopRight(), mpo.tensorAt(optPos));
}

template<typename T, int _Rows, int _Cols>
Matrix<T,Dynamic,Dynamic> DMRGOptimizer<T,_Rows,_Cols>::moveRight()
{
  std::size_t optPos = this->getPosition();
  mps.leftNormalize(optPos);
  if (projectors.size() > 0)
  {
    for (std::size_t i = 0; i < projectors.size(); ++i)
      projectors[i]->moveBuffer();
    updateProjectionMatrix(); 
  }
  return mps.leftContractedTop(optPos, this->getTopLeft(), mpo.tensorAt(optPos));
}

template<typename T, int _Rows, int _Cols>
double DMRGOptimizer<T,_Rows,_Cols>::optimizeEnergy(double tol, int ncv) 
{
  double en = _optimizeEnergy<_Rows,_Cols>(*this, tol, ncv);
  // printf("At site %lu: en=%e\n", this->getPosition(), en);
  return en;
}

template<typename T, int _Rows, int _Cols>
double DMRGOptimizer<T,_Rows,_Cols>::fullSweep(
    const std::vector<double>& tols, int ncv)
{
  double en;
  for (std::size_t i = 0; i < mps.N-1; ++i)
  {
    en = optimizeEnergy(tols[i], ncv);
    //std::cout << "At site " << optPos << ": en = " << en << std::endl;
    // moveRight();
    this->moveBuffer();
  }
  for (std::size_t i = mps.N-1; i > 0; --i)
  {
    en = optimizeEnergy(tols[i], ncv);
    //std::cout << "At site " << optPos << ": en = " << en << std::endl;
    // moveLeft();
    this->moveBuffer();
  }
  return en;
}

template<typename T, int _Rows, int _Cols>
void DMRGOptimizer<T,_Rows,_Cols>::applyQuadraticForm(T* vin, T* vout)
{
  std::size_t Dl, Dr, d;
  std::size_t optPos = this->getPosition();
  mps.getBondDimensions(optPos, Dl, Dr);
  d = mps.getPhysDim(optPos); 
  Vec tmp(d*Dl*Dr);
  applyQuadraticForm(vin, vout, tmp); 
}

template<typename T, int _Rows, int _Cols>
void DMRGOptimizer<T,_Rows,_Cols>::applyQuadraticForm(T* vin, T* vout, Vec& tmp)
{
  Vec vinProjected;
  const Mat& mL = this->getTopLeft(); 
  const Mat& mR = this->getTopRight();
  std::size_t optPos = this->getPosition();
  const MPOTensor<T, _Rows, _Cols>& mW = mpo.tensorAt(optPos);
  std::size_t Dl, Dr, d;
  mps.getBondDimensions(optPos, Dl, Dr);
  d = mps.getPhysDim(optPos); 

  if (projectors.size() > 0)
  {
    vinProjected = Map<const Vec, Aligned>(vin, d*Dl*Dr);
    //std::cout << "projectionMatrix has shape (" << projectionMatrix.rows() 
    //   << "," << projectionMatrix.cols() << ")" << std::endl; 
    vinProjected -= projectionMatrix 
      * (projectionMatrix.adjoint()*Map<const Vec, Aligned>(vin, d*Dl*Dr)).eval();
    vin = vinProjected.data();
  } 
  // std::cout << "At pos " << optPos << std::endl; 
  // std::cout << "mL.shape(): " << mL.rows() << "," << mL.cols() << std::endl; 
  // std::cout << "mR.shape(): " << mR.rows() << "," << mR.cols() << std::endl; 
  // std::cout << "mW.shape(): " << mW.Dl << "," << mW.Dr << std::endl;

  //for (std::size_t i = 0; i < d*Dl*Dr; ++i)
  //  vout[i] = 0;
  bool initialRun = true;
  for (std::size_t b2 = 0; b2 < mW.getDr(); ++b2)
  {
    for (typename MPOTensor<T,_Rows,_Cols>::RowIterator it = mW.rowIterator(b2);
         it; ++it)
    {
      // tmp.setZero();

      std::size_t b1 = it.row();
      const Matrix<T, _Rows, _Cols>& op = it.value();

      assert(mL.size() >= (b1+1)*Dl*Dl);
      assert(mR.size() >= (b2+1)*Dr*Dr);
      
      const T* rhoLp = mL.data()+b1*Dl*Dl;
      const T* rhoRp = mR.data()+b2*Dr*Dr;
      char noTrans[2] = "N";
      T alpha = 1;
      T beta = 0;
      int m = Dl;
      int n = d*Dr; 
      int k = Dl;
      // lda corresponds to the stride between consecutive columns,
      // i.e. number of rows of A
      int lda = m;
      int ldb = k;
      int ldc = m;

      // contracts 
      // W_{sigma_s' sigma_s}^{b1 b2} L^{b1}_{a_{s-1}' a_{s-1}} 
      // A_{a_{s-1} sigma_s a_s}
      // and stores the result in tmp
      gemm(noTrans, noTrans, &m, &n, &k, &alpha, rhoLp, &lda, vin, &ldb,
           &beta, tmp.data(), &ldc);
      if (! it.isIdentity())
        mps.applyOp(tmp.data(), op, Dl, Dr);
     
      alpha = 1;
      // if this is the first run, overwrite previous data in vout, otherwise
      // add
      beta = initialRun ? 0 : 1;
      m = Dl*d;
      n = Dr;
      k = Dr;
      lda = m;
      ldb = k;
      ldc = m;
      
      // contracts tmp_{a_{s-1}' sigma_s' a_s} R^{b2}_{a_s a_s'} and 
      gemm(noTrans, noTrans, &m, &n, &k, &alpha, tmp.data(), &lda, rhoRp, &ldb,
           &beta, vout, &ldc);
      initialRun = false;

      /* previous code
      // are those aligned?
      // assumes column-major order
      Map<const Mat, Aligned> rhoL(mL.data()+b1*Dl*Dl, Dl, Dl);
      Map<const Mat, Aligned> rhoR(mR.data()+b2*Dr*Dr, Dr, Dr);
      
      Map<Mat, Aligned> tmpMap1(tmp.data(), Dl, d*Dr); 
      tmpMap1.noalias() = rhoL*Map<const Mat, Aligned>(vin, Dl, d*Dr);
      mps.applyOp(tmp.data(), op, Dl, Dr);
      
      Map<Mat, Aligned> tmpMap2(tmp.data(), Dl*d, Dr);
      res.noalias() += tmpMap2*rhoR;
      */
    }
  }
  if (initialRun)
  { 
    // strange case: no contribution at all at current site
    // just make sure that vout is zero
    for (std::size_t i = 0; i < d*Dl*Dr; ++i)
      vout[i] = 0;
  }
  else if (projectors.size() > 0)
  {
    Map<Vec, Aligned> voutMap(vout, d*Dl*Dr);
    voutMap -= projectionMatrix*(projectionMatrix.adjoint()*voutMap).eval();
  }
  ++callCount; 
}






template <int _Rows, int _Cols>
double _optimizeEnergy(DMRGOptimizer<double, _Rows, _Cols>& dmrg, double tol, 
                       int ncv)
{
  MPS<double>& mps = dmrg.mps;
  std::size_t optPos = dmrg.getPosition();
  std::size_t d, Dl, Dr;
  d = mps.getPhysDim(optPos);
  mps.getBondDimensions(optPos, Dl, Dr);
  VectorXd& v0 = mps.tensorAt(optPos);

  //size of matrix n
  int n= d*Dl*Dr;
  // number of eigenvalues
  int nev= 1;

  //std::cout << "tol: " << tol << std::endl;
  std::vector< double > eigenvals(nev);
  std::vector< Matrix<double, Dynamic, 1> > eigenvecs(nev, 
      Matrix<double, Dynamic, 1>(d*Dl*Dr, 1) );

  int ido= 0; /* Initialization of the reverse communication
                 parameter. */

  char bmat[2]= "I"; /* Specifies that the right hand side matrix
                        should be the identity matrix; this makes
                        the problem a standard eigenvalue problem.
                        Setting bmat = "G" would have us solve the
                        problem Av = lBv (this would involve using
                        some other programs from BLAS, however). */

  /* Selects the nev eigenvectors to compute. Possible options are
   *  LM: largest magnitude
   *  SM: smallest magnitude
   *  LA: largest real component
   */
  char which[3]= "SA"; 
 
  //double tol= 0.; /* Sets the tolerance; tol<=0 specifies 
  //                   machine precision */

  double* resid;
  resid= new double[n];
  // copy initial vector
  for (std::size_t i = 0; i < v0.size(); ++i)
    resid[i] = v0(i);

  /* The largest number of basis vectors that will be used in the 
   * Implicitly Restarted Arnoldi Process.  Work per major iteration is
   * proportional to N*NCV*NCV. */
  if (ncv == 0)
    ncv= 4*nev; 
  if (ncv > n) 
    ncv = n;

  double *v;
  int ldv= n;
  v= new double[ldv*ncv];

  int* iparam;
  iparam= new int[11]; /* An array used to pass information to the routines
                          about their functional modes. */
  iparam[0]= 1;   // Specifies the shift strategy (1->exact)
  iparam[2]= 3*n; // Maximum number of iterations
  iparam[6]= 1;   /* Sets the mode of dsaupd.
                     1 is exact shifting,
                     2 is user-supplied shifts,
                     3 is shift-invert mode,
                     4 is buckling mode,
                     5 is Cayley mode. */

  int* ipntr;
  ipntr= new int[11]; /* Indicates the locations in the work array workd
                         where the input and output vectors in the
                         callback routine are located. */

  double* workd;
  workd= new double[3*n];

  double* workl;
  workl= new double[ncv*(ncv+8)];

  int lworkl= ncv*(ncv+8); /* Length of the workl array */

  int info= 1; /* Info=1 specifies that resid contains the initial guess 
                  to be used. */

  int rvec= 1; /* Specifies that eigenvectors should be calculated */

  int* select;
  select= new int[ncv];
  double* dvals;
  dvals = new double[2*ncv]; /* This vector will return the eigenvalues from
                           the second routine, dseupd. */
  double sigma;
  int ierr;

  /* Here we enter the main loop where the calculations are
     performed.  The communication parameter ido tells us when
     the desired tolerance is reached, and at that point we exit
     and extract the solutions. */
  Matrix<double, Dynamic, 1> tmp(d*Dl*Dr);
  do{
    dsaupd_(&ido, bmat, &n, which, &nev, &tol, resid, 
        &ncv, v, &ldv, iparam, ipntr, workd, workl,
        &lworkl, &info);

    if (ido==1 || ido==-1) 
      dmrg.applyQuadraticForm(workd+ipntr[0]-1, workd+ipntr[1]-1, tmp);
  }while (ido==1 || ido==-1);

  /* From those results, the eigenvalues and vectors are
     extracted. */

  if (info<0) {
    std::cout << "Error with dsaupd, info = " << info << "\n";
    std::cout << "Check documentation in dsaupd\n\n";
  } else {
    char selstr[4] = "All";
    dseupd_(&rvec, selstr, select, dvals, v, &ldv, &sigma, bmat,
        &n, which, &nev, &tol, resid, &ncv, v, &ldv,
        iparam, ipntr, workd, workl, &lworkl, &ierr);

    if (ierr!=0) {
      std::cout << "Error with dseupd, info = " << ierr << "\n";
      std::cout << "Check the documentation of dseupd.\n\n";
    } else if (info==1) {
      std::cout << "Maximum number of iterations reached.\n\n";
    } else if (info==3) {
      std::cout << "No shifts could be applied during implicit\n";
      std::cout << "Arnoldi update, try increasing NCV.\n\n";
    }

    /* Before exiting, we copy the solution information over to
        the arrays of the calling program, then clean up the
        memory used by this routine.  For some reason, when I
        don't find the eigenvectors I need to reverse the order of
        the values. */
    for (std::size_t i=0; i<nev; i++)
      eigenvals[i] = dvals[nev-i-1];
    for (int i= 0; i<nev; ++i)
    {
      Matrix<double, Dynamic, 1>& vec = eigenvecs[i]; 
      for (int j= 0; j<n; ++j)  
        vec(j) = v[i*n+j];
    }
    mps.updateTensor(optPos, eigenvecs[0]);
  } 
  delete [] resid;
  delete [] v;
  delete [] iparam;
  delete [] ipntr;
  delete [] workd;
  delete [] workl;
  delete [] select;
  delete [] dvals;

  return eigenvals[0];
}

template <int _Rows, int _Cols>
double _optimizeEnergy(DMRGOptimizer<c_double, _Rows, _Cols>& dmrg, 
                       double tol, int ncv)
{
  MPS<c_double>& mps = dmrg.mps;
  std::size_t d, Dl, Dr;
  std::size_t optPos = dmrg.getPosition();
  d = mps.getPhysDim(optPos);
  mps.getBondDimensions(optPos, Dl, Dr);
  VectorXcd& v0 = mps.tensorAt(optPos);

  /* reverse communication parameter, must be zero for first call. */
  int ido = 0; 

  /* Specifies that the right hand side matrix should be the identity matrix;
   * this makes the problem a standard eigenvalue problem.  Setting bmat = "G"
   * would have us solve the problem Av = lBv (this would involve using some
   * other programs from BLAS, however). */
  char bmat[2] = "I"; 

  /* dimension of the input problem */
  int n = d*Dl*Dr;

  /* Selects the nev eigenvectors to compute. Possible options are
   *  LM: largest magnitude
   *  SM: smallest magnitude
   *  LR: largest real component
   *  SR: smallest real compoent
   *  LI: largest imaginary component
   *  SI: smallest imaginary component */
  char which[3] = "SR"; 
  
  /* number of eigenvectors to compute */
  int nev = 1;

  /* contains the intial guess eigenvector if info=1 */
  c_double *resid;
  resid = new c_double[n];
  for (int i = 0; i < v0.size(); ++i)
    resid[i] = v0(i);

  /* the largest number of basis vectors that will be used in the Implicitly 
   * Restarted Arnoldi Process.  Work per major iteration is proportional to 
   * N*NCV*NCV. */
  if (ncv == 0)
    ncv= 4*nev;  // scipy chooses 2*nev+1; 
  if (ncv>n) 
    ncv = n;
  
  /* containst the set of Arnoldi vectors */
  c_double *v;
  int ldv = n;
  v = new c_double[ldv*ncv];

  /* further parameters for the iteration */
  int *iparam;
  int maxiter = 3*n; // is set to 10*n in scipy code
  iparam = new int[11]; 
  iparam[0] = 1;   // Specifies the shift strategy (1->exact)
  iparam[2] = maxiter; // Maximum number of iterations
  iparam[3] = 1; // block size, must be set to 1
  iparam[6] = 1; // 1 for normal eigenvalue problem 

  /* Pointer to mark the starting locations in the WORKD and WORKL arrays
   * for matrices/vectors used by the Arnoldi iteration.  */
  int *ipntr; 
  ipntr = new int[14]; 

  /* work arrays, workd must be 3n in size */
  c_double *workd;
  workd = new c_double[3*n];

  /* Length of the workl array, 3*ncv*ncv+5*ncv is minimal value */
  int lworkl = 3*ncv*ncv + 5*ncv;   
  c_double *workl;
  workl = new c_double[lworkl];

  double *rwork;
  rwork = new double[ncv];

  /* if info == 0, a randomly initial residual vector is used.
   * if info != 0, resid contains the initial residual vector, possibly from 
   * a previous run.
   * Error flag on output. */
  int info = 1; 

  /* Here we enter the main loop where the calculations are
     performed.  The communication parameter ido tells us when
     the desired tolerance is reached, and at that point we exit
     and extract the solutions. */
  Matrix<c_double, Dynamic, 1> tmp(d*Dl*Dr);
  do {
    znaupd_(&ido, bmat, &n, which, &nev, &tol, resid, 
        &ncv, v, &ldv, iparam, ipntr, workd, workl,
        &lworkl, rwork, &info);
    if ((ido==1)||(ido==-1)) {
      // do the work
      dmrg.applyQuadraticForm(workd+ipntr[0]-1, workd+ipntr[1]-1, tmp);
    }
  } while ((ido==1)||(ido==-1));

  if (info<0) {
    std::cerr << "Error with znaupd, info = " << info << "\n";
    std::cerr << "Check documentation in dsaupd\n\n";
    std::exit(1);
  }

  /* From those results, the eigenvalues and vectors are
     extracted. */
  
  /* Specifies that eigenvectors should be calculated */
  int rvec = 1;

  /* All nev eigenvectors shall be calculated */ 
  char howmny[2]= "A";

  /* used as internal workspace for howmny="A" */
  int *select;
  select = new int[ncv];

  /* array where eigenvalues are stored */ 
  c_double *dvals;
  dvals = new c_double[nev+1]; 
  
  /* as z parameter we just give the v array, ldz is ldv
   * this is possible since we don't need a schur basis */
  /* sigma is not referenced for mode=1 */ 
  c_double sigma;
 
  /* workspace */ 
  c_double *workev;
  workev = new c_double[2*ncv];

  zneupd_(&rvec, howmny, select, dvals, v, &ldv, &sigma, workev,
      bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv,
      iparam, ipntr, workd, workl, &lworkl, rwork, &info);

  if (info == -14) {
    std::cerr << "No eigenvectors to sufficient accuracy found while "
      << "optimizing at site " << optPos << std::endl;
  }
  if (info != 0)
  {
    std::cerr << "Error in zneupd: info = " << info << std::endl;
    VectorXcd res(d*Dl*Dr);
    dmrg.applyQuadraticForm(v0.data(), res.data(), tmp); 
    c_double en = v0.adjoint()*res;
    // std::cerr << "Calculated en: " << en << std::endl;
    return en.real();
  }

  /* Before exiting, we copy the solution information over to
     the arrays of the calling program, then clean up the
     memory used by this routine.  For some reason, when I
     don't find the eigenvectors I need to reverse the order of
     the values. */

  std::vector< c_double > eigenvals(nev);
  std::vector< Matrix<c_double, Dynamic, 1> > eigenvecs(nev, 
      Matrix<c_double, Dynamic, 1>(d*Dl*Dr) );
  for (int i= 0; i<nev; ++i)
    eigenvals[i]= dvals[i];
 
  for (int i= 0; i<nev; ++i)
  {
    Matrix<c_double, Dynamic, 1>& vec = eigenvecs[i]; 
    for (int j= 0; j<n; ++j)  
      vec(j) = v[i*n+j];
  }
  delete[] resid;
  delete[] v;
  delete[] iparam;
  delete[] ipntr;
  delete[] workd;
  delete[] workl;
  delete[] rwork;
  delete[] dvals;
  delete[] select;
  delete[] workev;

  // std::cout << "eigenvals[0]: " << eigenvals[0] << std::endl;
  mps.updateTensor(optPos, eigenvecs[0]);

  return eigenvals[0].real();
}





#endif
