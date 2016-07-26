#ifndef __MPS_H
#define __MPS_H

#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <assert.h>
#include <cmath>
#include <numeric>

#include "mpo.h"
#include "svd.h"

using namespace Eigen;

/**
 * A class representing a matrix product state with a fixed number of sites.
 */
template<typename T>
class MPS
{
  typedef Matrix<T, Dynamic, 1> Vec;
  typedef Matrix<T, Dynamic, Dynamic> Mat;
public:
  MPS(const std::vector<std::size_t>& dims, std::size_t D);
  ~MPS();

  /**
   * Computes the norm of this %MPS.
   */
  T norm() const;
  
  Mat leftContractedTop(std::size_t s) const;

  /**
   * @see leftContractedTop(std::size_t, const MatrixBase<Derived>&, const
   * MatrixBase<OtherDerived>&) const
   */
  template<typename Derived>
  Mat leftContractedTop(std::size_t s, const MatrixBase<Derived>& rho) const;


  /**
   * Returns a new transfer operator \f$ \rho'_{a_{s}' a_{s}} = O_{\sigma_s'
   * \sigma_s} A_{a_{s-1}' a_s'}^{\sigma_s'*} \rho_{a_{s-1}' a_{s-1}}
   * A_{a_{s-1} a_s}^{\sigma_s}  \f$, where \f$A_{a_{s-1} a_s}^{\sigma_s}\f$ is
   * the tensor at position s, \f$\rho_{a_{s-1}' a_{s-1}}\f$ the transfer
   * operator rho and \f$O_{\sigma_s' \sigma_s}\f$ the operator op.
   * */
  template<typename Derived, typename OtherDerived>
  Mat leftContractedTop(std::size_t s, const MatrixBase<Derived>& rho,
      const MatrixBase<OtherDerived>& op) const;

  template<typename Derived>
  Mat leftContractedTop(
      std::size_t d, std::size_t Dl, std::size_t Dr,
      const Mat& bra, 
      const Mat& ket, 
      const MatrixBase<Derived>& rho) const;

  Mat rightContractedTop(std::size_t s) const;

  template<typename Derived>
  Mat rightContractedTop(std::size_t s, const MatrixBase<Derived>& rho) const;

  template<typename Derived>
  Mat rightContractedTop(
      std::size_t d, std::size_t Dl, std::size_t Dr,
      const Mat& bra, 
      const Mat& ket, 
      const MatrixBase<Derived>& rho) const;

  /**
   * Returns a new transfer operator \f$ \rho'_{a_{s-1} a_{s-1}'} =
   * O_{\sigma_s' \sigma_s} A_{a_{s-1} a_s}^{\sigma_s} \rho_{a_{s} a_{s}'}
   * A_{a_{s-1}' a_s'}^{\sigma_s'*}\f$, where \f$A_{a_{s-1} a_s}^{\sigma_s}\f$
   * is the tensor at position s, \f$\rho_{a_{s} a_{s}'}\f$ the transfer
   * operator rho and \f$O_{\sigma_s' \sigma_s}\f$ the operator op.   
   */
  template<typename Derived, typename OtherDerived>
  Mat rightContractedTop(std::size_t s, const MatrixBase<Derived>& rho,
      const MatrixBase<OtherDerived>& op) const;

  /**
   * Replaces the tensor \f$A_{a_{s-1} a_s}^{\sigma_s}\f$ at position s=pos
   * having dimensions \f$(D_l, d, D_r)\f$ with the tensor vec having the 
   * same dimension and data elements which are stored in column-major order.
   */
  inline void updateTensor(std::size_t pos, const Vec& vec) { tensors[pos] = vec; }

  /**
   * Computes the expectation value of the matrix product operator op.
   */
  template<int _Rows, int _Cols>
  T expValue(const MPO<T, _Rows, _Cols>& op) const;

  /**
   * Returns a new transfer operator \f$ \rho'_{(a_s' a_s) b_s} = A_{a_{s-1}
   * a_s}^{\sigma_s} A_{a_{s-1}' a_s'}^{\sigma_s'*} \rho_{(a_{s-1}'
   * a_{s-1})b_{s-1}} W_{b_{s-1} b_s}^{\sigma_s \sigma_s'}\f$ where
   * \f$\rho_{(a_{s-1}' a_{s-1}) b_{s-1}}\f$ is the transfer operator rho and
   * \f$W_{b_{s-1} b_s}^{\sigma_s \sigma_s'}\f$ the MPOTensor op.
   */
  template<int _Rows, int _Cols>
  Mat leftContractedTop(std::size_t s, const Mat& rho,
                        const MPOTensor<T, _Rows, _Cols>& op) const;

  /**
   * Returns a new transfer operator \f$ \rho'_{(a_{s-1} a_{s-1}') b_{s-1}} = A_{a_{s-1}
   * a_s}^{\sigma_s} A_{a_{s-1}' a_s'}^{\sigma_s'*} \rho_{(a_s
   * a_s')b_s} W_{b_{s-1} b_s}^{\sigma_s \sigma_s'}\f$ where
   * \f$\rho_{(a_s a_s') b_s}\f$ is the transfer operator rho and
   * \f$W_{b_{s-1} b_s}^{\sigma_s \sigma_s'}\f$ the MPOTensor op. 
   */
  template<int _Rows, int _Cols>
  Mat rightContractedTop(std::size_t s, const Mat& rho,
                        const MPOTensor<T, _Rows, _Cols>& op) const;

  /**
   * Left-normalizes the tensor at site s.
   */
  void leftNormalize(std::size_t s);

  /**
   * Right-normalizes the tensor at site s.
   */
  void rightNormalize(std::size_t s);

  void getBondDimensions(std::size_t pos, std::size_t& Dl,
                         std::size_t& Dr) const;

  inline std::size_t getPhysDim(std::size_t pos) const { return dims[pos]; }
  inline Vec& tensorAt(std::size_t pos) { return tensors[pos]; }
  inline const Vec& tensorAt(std::size_t pos) const { return tensors[pos]; }
  
  /**
   * Applies the \f$d \times d\f$ operator op to the tensor \f$A_{a_{s-1}
   * a_s}^{\sigma_s}\f$ with dimensions \f$(D_l, d, D_r)\f$, which is stored in
   * column-major ordering at the location tensor.  Result is \f$O_{\sigma_s'
   * \sigma_s} A_{a_{s-1} a_s}^{\sigma_s}\f$.
   */
  template<typename Derived>
  void applyOp(T* tensor, const MatrixBase<Derived>& op, std::size_t Dl, std::size_t Dr) const;

  /**
   * The number of sites/tensors in the matrix product state.
   */
  std::size_t N;
private:
  std::size_t linIndex(std::size_t d, std::size_t Dl, std::size_t Dr,
                       std::size_t alpha, std::size_t sigma, std::size_t beta) const
  {
    //#if MPS_STORDER == ColMajor
    std::size_t index = beta*(d*Dl) + sigma*Dl+alpha;
    assert(index < d*Dl*Dr);
    return index;
    //#elif MPS_STORDER == RowMajor
    //  return alpha*(d*Dr) + sigma*Dr + beta;
    //#endif
  }


  std::vector<Vec> tensors;
  std::vector<std::size_t> dims;
  std::size_t D;
};

/* different normalization factors are needed for complex and real datatypes.
 * this is a hack to get the correct factors. */

template<typename T> 
double getMPSNormalizationFactor();

template<>
double getMPSNormalizationFactor<double>() { return 12; }

template<>
double getMPSNormalizationFactor<c_double>() { return 6; }


template<typename T>
template<int _Rows, int _Cols>
T MPS<T>::expValue(const MPO<T, _Rows, _Cols>& mpo) const
{
  Mat rho = Mat::Ones(1, 1);
  for (std::size_t i = 0; i < N; ++i) {
    const MPOTensor<T, _Rows, _Cols>& mW = mpo.tensorAt(i);
    rho = leftContractedTop(i, rho, mW);
  }
  assert(rho.size() == 1);
  return rho(0,0); 
}

template<typename T>
void MPS<T>::leftNormalize(std::size_t s)
{
  assert(s < N-1);
  std::size_t Dl, Dr, d;
  d = dims[s];
  getBondDimensions(s, Dl, Dr);

  Map<Mat, Aligned> m(tensors[s].data(), Dl*d, Dr);

  Mat mU, mVt;
  VectorXd mS;
  svd(m, mU, mS, mVt);

  std::size_t Dl1, Dr1;
  getBondDimensions(s+1, Dl1, Dr1);
  assert(tensors[s+1].size() == d*Dl1*Dr1);

  // matrix multiplication does explicitly account for aliasing
  Map<Mat, Aligned>(tensors[s+1].data(), Dl1, d*Dr1) = \
    mS.asDiagonal()*mVt*Map<Mat, Aligned>(tensors[s+1].data(), Dl1, d*Dr1);
  assert(mU.size() == d*Dl*Dr);
  tensors[s] = Map<const Vec, Aligned>(mU.data(), mU.size());
}

template<typename T>
void MPS<T>::rightNormalize(std::size_t s)
{
  std::size_t Dl, Dr, d;
  d = dims[s];
  getBondDimensions(s, Dl, Dr);

  Map<Mat, Aligned> m(tensors[s].data(), Dl, Dr*d);

  Mat mU, mVt;
  VectorXd mS;
  svd(m, mU, mS, mVt);

  std::size_t Dl1, Dr1;
  getBondDimensions(s-1, Dl1, Dr1);

  Map<Mat, Aligned>(tensors[s-1].data(), Dl1*d, Dr1) =
    Map<Mat, Aligned>(tensors[s-1].data(), Dl1*d, Dr1)*mU*mS.asDiagonal();
  tensors[s] = Map<const Vec, Aligned>(mVt.data(), mVt.size());
}


template<typename T>
T MPS<T>::norm() const
{
  Mat rho = Mat::Ones(1, 1);
  for (std::size_t i = 0; i < N; ++i)
  {
    rho = leftContractedTop(i, rho);
  }
  assert(rho.size() == 1);
  return rho(0, 0);
}

template<typename T>
template<typename Derived>
void MPS<T>::applyOp(T* tensor, const MatrixBase<Derived>& op, std::size_t Dl,
                     std::size_t Dr) const
{
  assert(op.rows() == op.cols());
  std::size_t d = op.rows();
  //std::cout << "applyOp, op.rows(): " << d << std::endl;
  std::vector<T> tmp(d);
  // T tmpVal;
  // the following order assumes column-major ordering corresponding to 
  // the eigen and fortran default, i.e., the left-most index varies
  // fastest 
  for (std::size_t beta = 0; beta < Dr; ++beta)
  {
    for (std::size_t alpha = 0; alpha < Dl; ++alpha)
    {
      for (std::size_t i = 0; i < d; ++i)
      {
        tmp[i] = tensor[linIndex(d,Dl,Dr,alpha,i,beta)];
        // std::cout << "tmp[i]: " << tmp[i] << std::endl;
        T res = 0;
        for (std::size_t j = 0; j <= i; ++j)
          res += op(i,j)*tmp[j];
        for (std::size_t j = i+1; j < d; ++j)
        {
          res += op(i,j)*tensor[linIndex(d,Dl,Dr,alpha,j,beta)];
	}
        // std::cout << "res: "<< res << std::endl;
	tensor[linIndex(d,Dl,Dr,alpha,i,beta)] = res;
      }
    }
  }
}

template<typename T>
template<int _Rows, int _Cols>
Matrix<T, Dynamic, Dynamic> MPS<T>::leftContractedTop(
  std::size_t s, const Mat& rho, const MPOTensor<T, _Rows, _Cols>& mW) const
{
  std::size_t Dl, Dr, d;
  d = dims[s];
  getBondDimensions(s, Dl, Dr);

  // assume that rho has indices (a_{s-1}' a_{s-1}) b_{s-1}
  // const MPOTensor<T, _Rows, _Cols>& mW = mpo.tensorAt(s);
  Mat res = Mat::Zero(Dr*Dr, mW.getDr());
  Mat tmp(Dl, d*Dr);
  Mat mX(Dl*d, Dr);

  //std::cout << "leftContractedTop at site " << s << std::endl;
  //std::cout << "d, Dl, Dr: " << d << "," << Dl << "," << Dr << std::endl;
  //std::cout << "W.Dl, W.Dr: " << mW.Dl << "," << mW.Dr << std::endl;

  // what we would denote mW_{b_{s-1} b_s} is here mW_{b1 b2}
  for (std::size_t b2 = 0; b2 < mW.getDr(); ++b2)
  {
    // create matrix mX^{b_s}_{a_{s-1}' sigm_s'a_s}
    mX.setZero();
    for (typename MPOTensor<T,_Rows,_Cols>::RowIterator it = mW.rowIterator(b2); 
         it; ++it)
    {
      std::size_t b1 = it.row();
      const Matrix<T, _Rows, _Cols>& op = it.value();
      // gives rho_{a_{s-1}' a_{s-1}}^{b_{s-1}}
      assert(rho.col(b1).size() == Dl*Dl);
      Map<const Mat> currentRho(rho.col(b1).data(), Dl, Dl);
      // std::cout << "currentRho: " << currentRho << std::endl;
      // std::cout << "currentRho.size(): " << currentRho.size() << std::endl;
      // apply currentRho to tensor at s
      // results in tmp_{(a_{s-1}' (sigma a_s)}
      assert(tensors[s].size() == Dl*d*Dr);
      tmp.noalias() = currentRho*Map<const Mat, Aligned>(tensors[s].data(), Dl, d*Dr);
      // std::cout << "tmp: " << tmp << std::endl;
      // apply W_{sigma_s' sigma_s}^{b_{s-1} b_s} in place
      //std::cout << "applying op: " << op << std::endl;
      applyOp(tmp.data(), op, Dl, Dr);
      //std::cout << "tmp: " << tmp << std::endl;
      // std::cout << "tmp.size(): " << tmp.size() << std::endl;
      // tmp is still of form tmp_{a_{s-1}' (sigma' a_s)}
      // reshape into tmp_{(a_{s-1}' sigma') a_s}
      mX += Map<Mat, Aligned>(tmp.data(), Dl*d, Dr);
    }
    assert(res.col(b2).size() == Dr*Dr);
    Map<Mat> tmpMap(res.col(b2).data(), Dr, Dr);
    tmpMap.noalias() += \
      (Map<const Mat, Aligned>(tensors[s].data(), Dl*d, Dr).adjoint() * mX);
  }
  // std::cout << res << std::endl;
  return res;
}


template<typename T>
template<int _Rows, int _Cols>
Matrix<T, Dynamic, Dynamic> MPS<T>::rightContractedTop(
  std::size_t s, const Mat& rho, const MPOTensor<T, _Rows, _Cols>& mW) const
{
  std::size_t Dl, Dr, d;
  d = dims[s];
  getBondDimensions(s, Dl, Dr);

  // assume that rho has indices (a_{s+1} a_{s+1})' b_{s+1} 
  Mat res = Mat::Zero(Dl*Dl, mW.getDl());
  Mat tmp(Dl*d, Dr);
  Mat mX(Dl, d*Dr);
  // what we would denote mW_{b_{s-1} b_s} is here mW_{b1 b2}
  for (std::size_t b1 = 0; b1 < mW.getDl(); ++b1)
  {
    mX.setZero();
    for (typename MPOTensor<T,_Rows,_Cols>::ColIterator it = mW.colIterator(b1); 
         it; ++it)
    {
      std::size_t b2 = it.col();
      const Matrix<T, _Rows, _Cols>& op = it.value();
      // gives rho_{a_{s+1} a_{s+1}'}^{b_{s+1}}
      Map<const Mat> currentRho(rho.col(b2).data(), Dr, Dr);
      // apply currentRho to tensor at s
      tmp.noalias() = Map<const Mat>(tensors[s].data(), Dl*d, Dr)*currentRho;
      // apply W_{sigma_s' sigma_s}^{b_{s-1} b_s} in place
      applyOp(tmp.data(), op, Dl, Dr);
      // results in tmp_{(a_s sigma') a_{s+1}')}
      // add in the form tmp_{ a_s (sigma' a_{s+1}') }
      mX += Map<Mat>(tmp.data(), Dl, Dr*d);
    }
    Map<Mat> tmpMap(res.col(b1).data(), Dl, Dl);
    tmpMap.noalias() += \
      mX*Map<const Mat>(tensors[s].data(), Dl, d*Dr).adjoint(); 
  }
  return res;
}

template<typename T>
template<typename Derived, typename OtherDerived>
Matrix<T, Dynamic, Dynamic> MPS<T>::leftContractedTop(
    std::size_t s, 
    const MatrixBase<Derived>& rho, 
    const MatrixBase<OtherDerived>& op) const
{
  std::size_t Dl, Dr, d;
  d = dims[s];
  getBondDimensions(s, Dl, Dr);
  
  Mat tmp = rho * Map<const Mat, Aligned>(tensors[s].data(), Dl, d*Dr);
  applyOp(tmp.data(), op, Dl, Dr);
  tmp = Map<const Mat, Aligned>(tensors[s].data(), Dl*d, Dr).adjoint() \
        * Map<const Mat, Aligned>(tmp.data(), Dl*d, Dr);
  return tmp; 
}

template<typename T>
template<typename Derived>
Matrix<T, Dynamic, Dynamic> MPS<T>::leftContractedTop(
    std::size_t s, 
    const MatrixBase<Derived>& rho) const
{
  std::size_t Dl, Dr, d;
  d = dims[s];
  getBondDimensions(s, Dl, Dr);
 
  /* 
  Mat tmp = rho * Map<const Mat, Aligned>(tensors[s].data(), Dl, d*Dr);
  tmp = Map<const Mat, Aligned>(tensors[s].data(), Dl*d, Dr).adjoint() \
        * Map<const Mat, Aligned>(tmp.data(), Dl*d, Dr);
  return tmp; 
  */
  return leftContractedTop(d, Dl, Dr, tensors[s], tensors[s], rho);
}


template<typename T>
template<typename Derived>
Matrix<T,Dynamic, Dynamic> MPS<T>::leftContractedTop(
    std::size_t d, std::size_t Dl, std::size_t Dr,
    const Matrix<T, Dynamic, Dynamic>& bra, 
    const Matrix<T, Dynamic, Dynamic>& ket, 
    const MatrixBase<Derived>& rho) const
{
  Mat tmp = rho * Map<const Mat, Aligned>(ket.data(), Dl, d*Dr);
  tmp = Map<const Mat, Aligned>(bra.data(), Dl*d, Dr).adjoint() \
        * Map<const Mat, Aligned>(tmp.data(), Dl*d, Dr);
  return tmp; 
}

template<typename T>
Matrix<T, Dynamic, Dynamic> MPS<T>::leftContractedTop(std::size_t s) const
{
  std::size_t Dl, Dr, d;
  d = dims[s];
  getBondDimensions(s, Dl, Dr);
  
  Mat tmp = Map<const Mat, Aligned>(tensors[s].data(), Dl*d, Dr).adjoint() \
        * Map<const Mat, Aligned>(tensors[s].data(), Dl*d, Dr);
  return tmp; 
}

template<typename T>
Matrix<T, Dynamic, Dynamic> MPS<T>::rightContractedTop(std::size_t s) const
{
  std::size_t Dl, Dr, d;
  d = dims[s];
  getBondDimensions(s, Dl, Dr);
  
  Mat tmp = Map<const Mat, Aligned>(tensors[s].data(), Dl, d*Dr) \
            * Map<const Mat, Aligned>(tensors[s].data(), Dl, Dr*d).adjoint();
  return tmp; 
}

template<typename T>
template<typename Derived>
Matrix<T, Dynamic, Dynamic> MPS<T>::rightContractedTop(
      std::size_t d, std::size_t Dl, std::size_t Dr,
      const Matrix<T, Dynamic, Dynamic>& bra, 
      const Matrix<T, Dynamic, Dynamic>& ket, 
      const MatrixBase<Derived>& rho) const
{ 
  Mat tmp = Map<const Mat, Aligned>(ket.data(), Dl*d, Dr)*rho;
  tmp = Map<const Mat, Aligned>(tmp.data(), Dl, Dr*d) \
        * Map<const Mat, Aligned>(bra.data(), Dl, Dr*d).adjoint();
  return tmp; 
}

template<typename T>
template<typename Derived>
Matrix<T, Dynamic, Dynamic> MPS<T>::rightContractedTop(
    std::size_t s, 
    const MatrixBase<Derived>& rho) const
{
  std::size_t Dl, Dr, d;
  d = dims[s];
  getBondDimensions(s, Dl, Dr);
  
  return rightContractedTop(d, Dl, Dr, tensors[s], tensors[s], rho); 
}

template<typename T>
template<typename Derived, typename OtherDerived>
Matrix<T, Dynamic, Dynamic> MPS<T>::rightContractedTop(
    std::size_t s, 
    const MatrixBase<Derived>& rho, 
    const MatrixBase<OtherDerived>& op) const
{
  std::size_t Dl, Dr, d;
  d = dims[s];
  getBondDimensions(s, Dl, Dr);
  
  Mat tmp = Map<const Mat, Aligned>(tensors[s].data(), Dl*d, Dr)*rho;
  applyOp(tmp.data(), op, Dl, Dr);
  tmp = Map<const Mat, Aligned>(tmp.data(), Dl, Dr*d) \
        * Map<const Mat, Aligned>(tensors[s].data(), Dl, Dr*d).adjoint();
  return tmp; 
}

template<typename T>
void MPS<T>::getBondDimensions(std::size_t pos, std::size_t& Dl,
                               std::size_t& Dr) const
{
  std::size_t currentD;

  if (pos < N/2)
  {
    // count from left end
    currentD = 1;
    for (std::size_t i = 0; i < pos; ++i)
    {
      currentD *= dims[i];
      if (currentD > D)
      {
        currentD = D;
        break;
      }
    }
    Dl = currentD;
    Dr = std::min(currentD*dims[pos], this->D);
  }
  else
  {
    // count from right end
    currentD = 1;
    for (std::size_t i = N-1; i > pos; --i)
    {
      currentD *= dims[i];
      if (currentD > D)
      {
        currentD = D;
        break;
      }
    }
    Dr = currentD;
    Dl = std::min(currentD*dims[pos], this->D);
    if (N % 2 != 0 && pos == N/2)
    {
      // N is odd and pos is pointing to the center of the
      // chain. Dl == Dr in this case
      Dl = Dr;
    }
  }
}


template<typename T>
MPS<T>::MPS(const std::vector<std::size_t>& dims_, std::size_t D_): \
  N(dims_.size()), dims(dims_), D(D_) 
{
  double prodd = 1;
  double prodD = 1;
  double tmpD = 1;
  double tmpd = 1;
  /* here we compute (d D)^(1/(2N)) by computing the product of d and D
   * up to size 1e30 and then taking the root in order to avoid problems
   * with numerical accuracy */
  for (std::size_t i = 0; i < N; ++i)
  {
    tmpd *= dims[i];
    if (tmpd > 1e30)
    {
      prodd *= std::pow(tmpd, 0.5/N);
      tmpd = 1;
    }
    std::size_t Dl, Dr;
    getBondDimensions(i, Dl, Dr);

    if (i < N-1)
    {
      tmpD *= Dr;
      if (tmpD > 1e30)
      {
        prodD *= std::pow(tmpD, 0.5/N);
        tmpD = 1;
      }
    }
  }
  prodD *= std::pow(tmpD, 0.5/N);
  prodd *= std::pow(tmpd, 0.5/N);
  // std::cout << "prodd: " << prodd << ", prodD: " << prodD << std::endl;
  // for real, we should have sqrt(12)
  double factor = std::sqrt(getMPSNormalizationFactor<T>())*1./(prodd*prodD);
  // std::cout << "davg: " << d_avg << std::endl;
  // std::cout << "factor: " << factor << std::endl;
  for (std::size_t i = 0; i < N; ++i)
  {
    std::size_t Dl, Dr, d;
    d = dims[i];
    getBondDimensions(i, Dl, Dr);

    //std::cout << "A[" << i << "].shape = (" << Dl << "," << d
    //          << "," << Dr << ")" << std::endl;

    Vec mA(d*Dl*Dr);
    // setRandom() sets the entries of mA to (possibly complex) numbers in the
    // interval [-1,1]
    mA.setRandom();
    mA.array() *= 0.5 * factor;
    tensors.push_back(mA);
  }
}


template<typename T>
MPS<T>::~MPS()
{
}

#endif


