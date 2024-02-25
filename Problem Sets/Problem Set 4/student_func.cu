//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */


__global__
void filter(
  unsigned int *outputVals, unsigned int * const inputVals, int numElems, int numBins, unsigned int i, int b
) {
  unsigned int mask = (numBins - 1) << i;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < numElems) {
    unsigned int t = 1 - ((inputVals[j] & mask) >> i);  // 1 if bit is unset
    t = t - 2 * b * t + b;  // flips t if b is 1
    outputVals[j] = t;
  }
}

__global__
void scan(unsigned int *outputVals, unsigned int *filterOut, int numElems) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j == 0) {
    outputVals[0] = 0;
    for (int j = 1; j < numElems; ++j) {
      outputVals[j] = outputVals[j - 1] + filterOut[j - 1];
    }
  }
}

__global__
void compact(
  unsigned int *outputVals,
  unsigned int *outputPos,
  unsigned int *inputVals,
  unsigned int *inputPos,
  unsigned int *filterOut,
  unsigned int *scanOut,
  int numElems,
  int start
) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < numElems && filterOut[j]) {
      outputVals[start + scanOut[j]] = inputVals[j];
      outputPos[start + scanOut[j]] = inputPos[j];
  }
}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  int blockSize = 1024;
  int gridSize = ceil((float) numElems / blockSize);

  const int numBits = 1; // TODO: generalize
  const int numBins = 1 << numBits;

  unsigned int *d_vals_src = d_inputVals;
  unsigned int *d_pos_src  = d_inputPos;
  unsigned int *d_vals_dst = d_outputVals;
  unsigned int *d_pos_dst  = d_outputPos;

  unsigned int *d_filterOut, *d_scanOut;
  checkCudaErrors(cudaMalloc(&d_filterOut, numElems * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_scanOut, numElems * sizeof(unsigned int)));
  unsigned int *h_filterOutLast = new unsigned int[1];
  unsigned int *h_scanOutLast = new unsigned int[1];
  for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i += numBits) {
    unsigned int start = 0;
    for (int b = 0; b < numBins; ++b) {
      filter<<<gridSize, blockSize>>>(d_filterOut, d_vals_src, numElems, numBins, i, b);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
      checkCudaErrors(cudaMemcpy(h_filterOutLast, d_filterOut + numElems - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost));
      scan<<<gridSize, blockSize>>>(d_scanOut, d_filterOut, numElems);
      checkCudaErrors(cudaMemcpy(h_scanOutLast, d_scanOut + numElems - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost));
      compact<<<gridSize, blockSize>>>(d_vals_dst, d_pos_dst, d_vals_src, d_pos_src, d_filterOut, d_scanOut, numElems, start);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
      start += *h_scanOutLast + *h_filterOutLast;
    }

    std::swap(d_vals_dst, d_vals_src);
    std::swap(d_pos_dst, d_pos_src);
  }

  checkCudaErrors(cudaFree(d_filterOut));
  checkCudaErrors(cudaFree(d_scanOut));

  checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
}
