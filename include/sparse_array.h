#ifndef __SPARSE_ARRAY_H
#define __SPARSE_ARRAY_H

#include <vector>

/**
 * A sparse array with compressed column storage.
 */
template<typename T>
class SparseArray
{
public:
  SparseArray(std::size_t nrows, std::size_t ncols, T zeroVal=0.);

  class Iterator;
  class RowIterator;
  class ColIterator; 

  /**
   * Sets the element at position (row, col) to the value elem. 
   */
  void setElement(std::size_t row, std::size_t col, const T& elem);

  /**
   * Returns the element at position (row, col). Returns zeroVal if the element
   * has not been set to a different value through a call to setElement().
   * Note that this is inefficient since the whole array has to be searched.
   */
  T& getElement(std::size_t row, std::size_t col);

  /**
   * Returns the i-th stored element.
   */
  const T& getElement(std::size_t i) const { return data[i]; }

  /**
   * Returns a RowIterator iterating throught the rows in column col.
   * Note that this is efficient since columns are stored consecutively in
   * memory in a compressed column storage format.
   */
  RowIterator rowIterator(std::size_t col) const { return RowIterator(*this, col); }
  
  /**
   * Returns a ColIterator iterating throught the columns in row row.
   * Note that this is inefficient since rows are not stored consecutively in
   * memory in a compressed column storage format.
   */
  ColIterator colIterator(std::size_t row) const { return ColIterator(*this, row); }

  /**
   * Returns the row index of the element data[i].
   */
  std::size_t row(std::size_t i) const { return rowIndices[i]; }

  /**
   * Returns the total number of columns.
   */
  std::size_t cols() const { return ncols; }

  /**
   * Returns the total number of rows.
   */
  std::size_t rows() const { return nrows; }

  /**
   * Returns the total number of non-zero elements.
   */
  std::size_t nonZeros() const { return data.size(); }

  /**
   * Returns the number of non-zero elements in column col.
   */
  std::size_t nonZeros(std::size_t col) const
  {
    return colPtrs[col+1]-colPtrs[col];
  }

  std::size_t nrows;
  std::size_t ncols;
  T zeroVal;
  std::vector<T> data;

  /**
   * Stores the row indices of the elements in data such that the element at 
   * data[i] has a row index rowIndices[i].
   */
  std::vector<std::size_t> rowIndices;

  /**
   * colPtrs[col] is the index of the first element in the array data belonging
   * to column col. Consequently, colPtrs[col+1]-colPtrs[col] is the number of
   * elements in column col.
   */ 
  std::vector<std::size_t> colPtrs;
};

template<typename T>
SparseArray<T>::SparseArray(std::size_t nrows_, std::size_t ncols_,
                              T zeroVal_) : 
      nrows(nrows_), ncols(ncols_), zeroVal(zeroVal_),
      colPtrs(ncols_+1, 0)
{ 
}

template<typename T>
T& SparseArray<T>::getElement(std::size_t row, std::size_t col)
{
  std::size_t pos = colPtrs[col];
  while (pos < colPtrs[col+1])
  {
    if (rowIndices[pos] == row)
    {
       // element is found
       return data[pos];
    }
    else if (rowIndices[pos] > row)
    {
      // element has not been found.
      break;
    } 
    ++pos;
  }
  return zeroVal;
}

template<typename T>
void SparseArray<T>::setElement(
    std::size_t row, std::size_t col, const T& elem)
{
  // colPtrs[col] gives the index (in data) of the first element stored in 
  // column col
  // new element will be stored between colPtrs[col] and colPtrs[col+1]
  std::size_t pos = colPtrs[col];
  // loop through current column
  while (pos < colPtrs[col+1])
  {
    if (rowIndices[pos] == row)
    {
       // element is found
       data[pos] = elem;
       return;
    }
    else if (rowIndices[pos] > row)
    {
      // the element at position pos starts a new row, so we insert 
      // the element at pos. 
      break;
    } 
    ++pos;
  }

  rowIndices.insert(rowIndices.begin()+pos, row);
  data.insert(data.begin()+pos, elem);
 
  // the index of the colPtrs of all elements in higher columns must be
  // increased by one to account for addition of element
  for (std::size_t currentCol = col+1; currentCol < ncols+1; ++currentCol)
    ++colPtrs[currentCol];

}


template<typename T>
class SparseArray<T>::ColIterator
{
public:
  ColIterator(const SparseArray<T>& sparse_, std::size_t row_) : 
    sparse(sparse_), index(-1),
    rowIndex(row_),  colIndex(0)
  { nextIndex(0); }
  ColIterator(const RowIterator& it) : 
    sparse(it.sparse), index(it.index), 
    rowIndex(it.rowIndex), colIndex(it.colIndex)
  { }
   
  operator bool() const { return (index < sparse.data.size()); }

  inline const T& value() const { return sparse.data[index]; }
  inline std::size_t row() const { return rowIndex; }
  inline std::size_t col() const { return colIndex; }
  inline void nextIndex() { nextIndex(index+1); }
  inline void nextIndex(std::size_t i)
  {
    std::size_t j = (std::size_t) colIndex;
    
    for (; i < sparse.data.size(); ++i)
    {
      if (sparse.rowIndices[i] == (std::size_t) rowIndex)
      {
        break;
      }
    }
    for (; j < sparse.cols(); ++j)
    {
      if (sparse.colPtrs[j] > i)
      {
        break;
      }
    }

    index = i;
    colIndex = j-1;

  }
  inline ColIterator& operator++()
  {
    nextIndex();
    return *this;
  }
private:
  const SparseArray<T>& sparse;
  std::size_t index;
  std::size_t rowIndex;
  std::size_t colIndex;
};


template<typename T>
class SparseArray<T>::RowIterator 
{
public:
  RowIterator(const SparseArray<T>& sparse_, std::size_t col_) : 
    sparse(sparse_), index(sparse_.colPtrs[col_]), 
    endIndex(sparse_.colPtrs[col_+1]), colIndex(col_)
  { }
  RowIterator(const RowIterator& it) : 
    sparse(it.sparse), index(it.index), endIndex(it.endIndex), 
    colIndex(it.colIndex)
  { }
   
  operator bool() const { return (index < endIndex); }
  // inline T& value() { return sparse.data[index]; }
  inline const T& value() const { return sparse.data[index]; }
  inline std::size_t row() const { return sparse.rowIndices[index]; }
  inline std::size_t col() const { return colIndex; }
  inline RowIterator& operator++()
  {
    ++index;
    return *this;
  }
private:
  const SparseArray<T>& sparse;
  std::size_t index;
  std::size_t endIndex;
  std::size_t colIndex;
};


#endif /* __SPARSE_ARRAY_H */
