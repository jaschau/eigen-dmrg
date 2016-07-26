#ifndef __MPO_H
#define __MPO_H

#include "sparse_array.h"
#include <Eigen/Dense>

using namespace Eigen;

/**
 * A class representing the individual tensors of a matrix product operator. 
 *
 * In typical matrix product operators, the tensors have a sparse structure,
 * i.e., only few of the operator-valued elements are different from zero.  In
 * this implementation, the operator-valued entries are stored in a compressed
 * column sparse format. Note that this makes iterations through indiviual
 * columns efficient since the columns are stored consecutively in memory,
 * whereas iterating through rows is slightly less efficient.
 */
template<typename T, int _Rows, int _Cols> 
class MPOTensor 
{ 
public:
  MPOTensor(std::size_t Dl, std::size_t Dr); ~MPOTensor() {};

  /**
   * Wrapper class for stored matrices, storing flags like isIdentity in
   * addition to the actual matrix.
   */ 
  class Operator
  {
  public:
    bool isIdentity;
    Matrix<T, _Rows, _Cols> m;
  };

  /**
   * Wrapper iterator class which allows iterating through the rows and columns
   * of the underlying sparse data structure SparseArray through its iterator
   * classes RowIterator and ColIterator, while also giving access to the
   * additional stored flags like isIdentity. 
   */
  template <typename It>
  class Iterator
  {
  public:
    Iterator(const It& it_) : it(it_) {}
    std::size_t row() const { return it.row(); }
    std::size_t col() const { return it.col(); }
    const Matrix<T,_Rows,_Cols>& value() const { return it.value().m; }
    bool isIdentity() const { return it.value().isIdentity; }
    operator bool() const { return it; }
    inline Iterator& operator++() { ++it; return *this; }
  private:
    It it;
  };

  /**
   * Returns the number of left bonds, i.e., the number of rows of the tensor.
   */
  std::size_t getDl() const { return Dl; }

  /**
   * Returns the number of right bonds, i.e., the number of columns of the
   * tensor.
   */
  std::size_t getDr() const { return Dr; }

  /**
   * Sets the element at row i and column j to the matrix m. The flag
   * isIdentity indicates whether or not the matrix m is supposed to be
   * considered the identity matrix. This flag may be used by implementations
   * to increase efficiency.
   */
  void setElement(std::size_t i, std::size_t j, const Matrix<T,_Rows,_Cols>& m,
                  bool isIdentity=false)
  {
    Operator op;
    op.m = m;
    op.isIdentity = isIdentity;
    ops.setElement(i,j,op);
  }

  typedef MPOTensor<T,_Rows,_Cols>::Iterator< typename SparseArray<Operator>::RowIterator > RowIterator;
  typedef MPOTensor<T,_Rows,_Cols>::Iterator< typename SparseArray<Operator>::ColIterator > ColIterator;

  /**
   * Returns a RowIterator instance iterating through the rows of the column 
   * col.
   */
  RowIterator rowIterator(std::size_t col) const 
  { 
    return RowIterator(ops.rowIterator(col)); 
  };

  /**
   * Returns a ColIterator instance iterating through the columns of the row 
   * row.
   */ 
  ColIterator colIterator(std::size_t row) const 
  { 
    return ColIterator(ops.colIterator(row)); 
  };

  /**
   * Computes the Kronecker product with another MPOTensor mpo.  Denoting this
   * tensor of size \f$(D_l, D_r)\f$ by W and the tensor mpo of size \f$(D_l',
   * D_r')\f$ by W', the tensor product results in a new tensor W'' of size
   * \f$(D_l D_l', D_r D_r')\f$ with elements 
   * \f$ 
   *   {W''}_{b_l' + b_l D_l'; b_r' + b_r D_r'} = W_{b_l, b_r} {W'}_{b_l', b_r'}.  
   * \f$
   * Note that in a previous version, we computed
   * \f$ 
   *   {W''}_{b_l + b_l' D_l; b_r + b_r' D_r} = W_{b_l, b_r} {W'}_{b_l', b_r'}, 
   * \f$
   * which does not correspond to the conventional definition of the Kronecker
   * product.
   * TODO: check that the change does not have unwanted side effects. 
   */
  MPOTensor<T, _Rows, _Cols> kroneckerProduct(const MPOTensor<T, _Rows, _Cols>& mpo) const;
  
private:
  SparseArray< Operator > ops;
  std::size_t Dl;
  std::size_t Dr;
};


template<typename T, int _Rows, int _Cols>
class MPO
{
  typedef Matrix<T, _Rows, _Cols> Mat;

public:
  MPO(std::size_t N);

  // virtual MPO<T,_Rows,_Cols> dot(const MPO<T,_Rows,_Cols>& mpo) const;
  virtual const MPOTensor<T, _Rows, _Cols>& tensorAt(std::size_t pos) const;
  std::size_t N;
private:
  MPOTensor<T, _Rows, _Cols> emptyTensor;
};

template<typename T, int _Rows, int _Cols>
class TransInvMPO : public MPO<T, _Rows, _Cols>
{
public:
  TransInvMPO(std::size_t N, std::size_t D, bool staggered=false);
  TransInvMPO(const TransInvMPO& mpo) :
    MPO<T,_Rows,_Cols>(mpo.N),
    staggered(mpo.staggered), 
    D(mpo.D), wLeft(mpo.wLeft), wBulk(mpo.wBulk), wBulkStaggered(mpo.wBulkStaggered),
    wRight(mpo.wRight) { }
  virtual const MPOTensor<T, _Rows, _Cols>& tensorAt(std::size_t pos) const;
  TransInvMPO<T,_Rows,_Cols> dot(const TransInvMPO<T,_Rows,_Cols>& mpo) const;

  MPOTensor<T,_Rows,_Cols>& getLeftTensor() { return wLeft; }
  MPOTensor<T,_Rows,_Cols>& getBulkTensor() { return wBulk; }
  MPOTensor<T,_Rows,_Cols>& getBulkStaggeredTensor() { return wBulkStaggered; }
  MPOTensor<T,_Rows,_Cols>& getRightTensor() { return wRight; }
protected:
  bool staggered;
  std::size_t D;
  MPOTensor<T, _Rows, _Cols> wLeft;
  MPOTensor<T, _Rows, _Cols> wBulk;
  MPOTensor<T, _Rows, _Cols> wBulkStaggered;
  MPOTensor<T, _Rows, _Cols> wRight;
};


template<typename T, int _Rows, int _Cols>
class TransInvSingleSiteMPO : public TransInvMPO<T,_Rows,_Cols>
{
public:
  TransInvSingleSiteMPO(const TransInvSingleSiteMPO& mpo) :
    TransInvMPO<T,_Rows,_Cols>(mpo) { }
  TransInvSingleSiteMPO(std::size_t N_, const Matrix<T,_Rows,_Cols>& op)
    : TransInvMPO<T,_Rows,_Cols>(N_, 2, false)
  {
    std::size_t d = op.rows();
    Matrix<T,_Rows,_Cols> eye = Matrix<T,_Rows,_Cols>::Identity(d,d);
    this->wBulk.setElement(0, 0, eye, true);
    this->wBulk.setElement(1, 0, op);
    this->wBulk.setElement(1, 1, eye, true);

    this->wLeft.setElement(0, 0, op);
    this->wLeft.setElement(0, 1, eye, true); 

    this->wRight.setElement(0, 0, eye, true);
    this->wRight.setElement(1, 0, op);
  }
  TransInvSingleSiteMPO(std::size_t N_, const Matrix<T,_Rows,_Cols>& op,
      const Matrix<T,_Rows,_Cols>& opStaggered)
    : TransInvMPO<T,_Rows,_Cols>(N_, 2, true)
  {
    std::size_t d = op.rows();
    Matrix<T,_Rows,_Cols> eye = Matrix<T,_Rows,_Cols>::Identity(d,d);
    this->wBulk.setElement(0, 0, eye, true);
    this->wBulk.setElement(1, 0, op);
    this->wBulk.setElement(1, 1, eye, true);
    this->wBulkStaggered.setElement(0, 0, eye, true);
    this->wBulkStaggered.setElement(1, 0, opStaggered);
    this->wBulkStaggered.setElement(1, 1, eye, true);

    this->wLeft.setElement(0, 0, op);
    this->wLeft.setElement(0, 1, eye, true); 

    this->wRight.setElement(0, 0, eye, true);
    this->wRight.setElement(1, 0, op);
  }
};

template<typename T, int _Rows, int _Cols>
MPOTensor<T,_Rows,_Cols>::MPOTensor(std::size_t pDl, std::size_t pDr) :
  ops(pDl, pDr, Operator()),
  Dl(pDl), Dr(pDr)
{

}

template<typename T, int _Rows, int _Cols>
MPOTensor<T, _Rows, _Cols> MPOTensor<T,_Rows,_Cols>::kroneckerProduct(
    const MPOTensor<T, _Rows, _Cols>& mpo) const
{
  std::size_t Dlnew = Dl*mpo.getDl();
  std::size_t Drnew = Dr*mpo.getDr();

  std::size_t Dl2 = mpo.getDl();
  std::size_t Dr2 = mpo.getDr();

  MPOTensor<T, _Rows, _Cols> res(Dlnew, Drnew);
  for (std::size_t b2r = 0; b2r < Dr2; ++b2r)
  {
    for (std::size_t b1r = 0; b1r < Dr; ++b1r)
    {
      // std::size_t br = b2r*Dr + b1r;
      // maybe rather do
      std::size_t br = b1r*Dr2 + b2r;
      for (typename MPOTensor<T,_Rows,_Cols>::RowIterator it2 = mpo.rowIterator(b2r); 
           it2; ++it2)
      {
        for (typename MPOTensor<T,_Rows,_Cols>::RowIterator it1 = rowIterator(b1r); 
             it1; ++it1)
        {
          // std::size_t bl = it2.row()*Dl + it1.row();
          // maybe rather do
          std::size_t bl = it1.row()*Dl2 + it2.row();
          Matrix<T, _Rows, _Cols> resOp = it1.value()*it2.value();
          res.setElement(bl, br, resOp, it1.isIdentity() && it2.isIdentity());
        }
      }
    }
  }
  return res;
}


template<typename T, int _Rows, int _Cols>
MPO<T, _Rows, _Cols>::MPO(std::size_t pN) : N(pN), emptyTensor(0,0)
{
  // std::cout << "MPO: N = " << N << std::endl;
}

template<typename T, int _Rows, int _Cols>
const MPOTensor<T, _Rows, _Cols>& MPO<T, _Rows, _Cols>::tensorAt(std::size_t pos) const
{
  return emptyTensor;
}

template<typename T, int _Rows, int _Cols>
TransInvMPO<T, _Rows, _Cols>::TransInvMPO(std::size_t N_, std::size_t D_, 
        bool staggered_) :
    MPO<T, _Rows, _Cols>(N_), 
    staggered(staggered_), D(D_),
    wLeft(1,D_), wBulk(D_,D_), wBulkStaggered(D_, D_), wRight(D_,1)
{
  // std::cout << "N: " << this->N << std::endl;
}

template<typename T, int _Rows, int _Cols>
const MPOTensor<T, _Rows, _Cols>& TransInvMPO<T, _Rows, _Cols>::tensorAt(std::size_t pos) const 
{
  if (pos == 0)
    return wLeft;
  else if (pos == this->N-1)
    return wRight;
  if (! staggered || (pos % 2) == 0)
    return wBulk;
  return wBulkStaggered;
}

template<typename T, int _Rows, int _Cols>
TransInvMPO<T, _Rows, _Cols> TransInvMPO<T, _Rows, _Cols>::dot(const TransInvMPO<T,_Rows,_Cols>& mpo) const
{
  // std::cout << "calculating squared ham in TransInvMPO::dot" << std::endl;
  TransInvMPO<T,_Rows,_Cols> res(this->N, this->D*mpo.D, 
      this->staggered || mpo.staggered);

  res.wLeft = this->wLeft.kroneckerProduct(mpo.wLeft);
  res.wBulk = this->wBulk.kroneckerProduct(mpo.wBulk);
  if (this->staggered && mpo.staggered)
    res.wBulkStaggered = this->wBulkStaggered.kroneckerProduct(mpo.wBulkStaggered);
  else if (this->staggered && ! mpo.staggered)
    res.wBulkStaggered = this->wBulkStaggered.kroneckerProduct(mpo.wBulk);
  else if (! this->staggered && mpo.staggered)
    res.wBulkStaggered = this->wBulk.kroneckerProduct(mpo.wBulkStaggered);
  res.wRight = this->wRight.kroneckerProduct(mpo.wRight);

  return res;
}

#endif

