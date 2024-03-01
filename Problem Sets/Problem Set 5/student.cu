/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/

#include <thrust/device_vector.h>

#include "utils.h"

struct satisfies_mask : thrust::unary_function<unsigned int, bool> {
  satisfies_mask(unsigned int mask, unsigned int invMask)
    : mask(mask), invMask(invMask) {}

  __host__ __device__
  bool operator()(const unsigned int &x) {
    return ((x & mask) == mask) & ((~x & invMask) == invMask);
  }

private:

  unsigned int mask;
  unsigned int invMask;
};

void partial_sort(const unsigned int* const d_inputVals,
                  unsigned int * const d_outputVals,
                  long * const d_binSizes,
                  const size_t numElems,
                  const size_t numBits,
                  const size_t maxVal)
{ 
  size_t numBins = 1 << numBits;
  size_t msb = floor(log2(maxVal));

  thrust::device_vector<unsigned int> d_vals_src(d_inputVals, d_inputVals + numElems);
  thrust::device_vector<unsigned int> d_vals_dst(numElems);
  
  thrust::device_vector<long> d_binSizeVec(numBins);
  d_binSizeVec[0] = 0;

  auto start = d_vals_dst.begin();
  decltype(start) prev;
  for (unsigned int i = 0; i < numBins; ++i) {
    unsigned int mask = i << (msb - numBits + 1);
    unsigned int invMask = (numBins - i - 1) << (msb - numBits + 1);
    prev = start;
    start = thrust::copy_if(d_vals_src.begin(), d_vals_src.end(), start, satisfies_mask(mask, invMask));
    d_binSizeVec[i] = start - prev;
  }

  thrust::copy(d_vals_dst.begin(), d_vals_dst.end(), d_outputVals);
  thrust::copy(d_binSizeVec.begin(), d_binSizeVec.end(), d_binSizes);
}

__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible

  //Although we provide only one kernel skeleton,
  //feel free to use more if it will help you
  //write faster code
  
  // TODO: compute local histograms from a single coarse bin associated with the current thread block
  // this means we need a single coarse bin to fit into a thread block

  uint gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < numVals) {
    atomicAdd(histo + vals[gid], 1);
  }
}

__global__
void combineHistos(
  unsigned int * const localHistos,
  unsigned int * const binSizes,
  unsigned int * const histo,
  int numVals,
  unsigned int *blockStarts,
  unsigned int *blockEnds
) {
  uint gid = blockIdx.x * blockDim.x + threadIdx.x;
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  //TODO Launch the yourHisto kernel

  //if you want to use/launch more than one kernel,
  //feel free

  uint blockSize = 512;  // TODO: determine from numBins
  uint gridSize = ceil(numElems / blockSize);

  unsigned int *localHistos;
  unsigned int *blockStarts;
  unsigned int *blockEnds;

  int numBits = 1;
  int numCoarseBins = 1 << numBits;
  int maxVal = numBins - 1;

  unsigned int *d_sorted;
  long *d_binSizes;
  checkCudaErrors(cudaMalloc(&d_sorted, numElems * sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_binSizes, numCoarseBins * sizeof(long)));

  partial_sort(d_vals, d_sorted, d_binSizes, numElems, numBits, maxVal);
  
  // unsigned int *h_sorted = (unsigned int *) malloc(numElems * sizeof(unsigned int));
  // checkCudaErrors(cudaMemcpy(h_sorted, d_sorted, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));
  // for (int i = 0; i < numElems; ++i) {
  //   if (i % 10 == 0) {
  //     getchar();
  //     printf("\n");
  //   }
  //   printf("%u\n", h_sorted[i]);
  // }

  long *h_binSizes = (long *) malloc(numCoarseBins * sizeof(long));
  checkCudaErrors(cudaMemcpy(h_binSizes, d_binSizes, numCoarseBins * sizeof(long), cudaMemcpyDeviceToHost));
  long total = 0;
  for (size_t i = 0; i < numCoarseBins; ++i) {
    total += h_binSizes[i];
    printf("h_binSizes[%u] = %ld\n", i, h_binSizes[i]);
  }
  printf("Total: %ld\n", total);

  yourHisto<<<gridSize, blockSize>>>(d_sorted, d_histo, numElems);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
