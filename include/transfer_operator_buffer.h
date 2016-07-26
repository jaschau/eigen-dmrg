#ifndef __TRANSFER_OPERATOR_BUFFER_H
#define __TRANSFER_OPERATOR_BUFFER_H

#include <Eigen/Dense>

using namespace Eigen;

/** 
 * Base class for buffering the transfer operators to the left and right of the
 * current position as one sweeps through a matrix product state. 
 */
template<typename T, class Derived>
class TransferOperatorBuffer 
{
  typedef Matrix<T, Dynamic, Dynamic> Mat;

public:
  TransferOperatorBuffer(std::size_t N, int direction);

  /**
   * Makes a move from position pos to position pos-1 and returns the
   * updated transfer operator to the right of the new position pos-1. 
   */
  Mat moveLeft();

  /**
   * Implements a move from position pos to position pos+1 and returns the
   * updated transfer operator to the left of the new position pos+1. 
   */
  Mat moveRight();

  /**
   * Moves the buffer by one position in the current direction specified by 
   * TransferOperatorBuffer::getDirection(). Note that this will reverse the direction by
   * one. 
   **/
  void moveBuffer();

  /**
   * Returns the current position.
   */
  std::size_t getPosition() const { return pos; }

  /**
   * Returns the current move direction which can be either +1 or -1. Moves to
   * the right correspond to +1 while moves to the left correspond to -1. 
   */
  int getDirection() const { return direction; }
  
  /**
   * Returns the buffered transfer operator to the left of the current
   * position.
   */
  const Mat& getTopLeft() 
  {
    // storage convention is such that topBuffer[pos] stores the transfer
    // operator the the left of pos
    return topBuffer[pos]; 
  }

  /**
   * Returns the buffered transfer operator the the right of the current
   * position.
   */
  const Mat& getTopRight() 
  { 
    return topBuffer[pos+2];
  }

private:
  /**
   * Size of the chain.
   */
  std::size_t N;

  /**
   * Direction of the current sweep, +1 for moves to the right and -1 for moves
   * to the left.
   */
  int direction;

  /**
   * Buffer for the transfer operators to the left and to the right of the
   * current position.
   */
  std::vector<Mat> topBuffer;

  /**
   * The current position.
   */
  std::size_t pos;
};

template<typename T, class Derived>
TransferOperatorBuffer<T, Derived>::TransferOperatorBuffer(std::size_t N_, int direction_) :
  N(N_), direction(direction_), topBuffer(N_+2)
{
  pos = direction == 1 ? 0 : N-1;

  topBuffer[N+1] = Mat::Ones(1,1);
  topBuffer[0] = Mat::Ones(1,1);
}

template<typename T, class Derived>
void TransferOperatorBuffer<T, Derived>::moveBuffer()
{
  if (direction == -1)
  {
    // we move left 
    if (pos <= 0) {
      std::cerr << "pos <= 0 in move" << std::endl;
      return;
    }
    
    topBuffer[pos+1] = static_cast<Derived*>(this)->moveLeft(); 

    --pos;
    if (pos == 0)
    {
      // optimizing at left-most point, reverse direction
      direction *= -1;
    }
  }
  else if (direction == 1)
  {
    if (pos >= N-1) {
      std::cerr << "pos >= 0 in move" << std::endl;
      return;
    } 
    // convention is such that topsBuffer[optPos] stores the L top for 
    // position optPos
    topBuffer[pos+1] = static_cast<Derived*>(this)->moveRight(); 

    ++pos;
    if (pos == N-1)
    {
      direction *= -1;
    }
  }
}

#endif
