/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include <functional>
#include "utils.h"
// #include "student_func.cuh"

template <typename T>
struct min_op {
   __device__
   T operator()(T x, T y) const {
      return min(x, y);
   }
};

template <typename T>
struct max_op {
   __device__
   T operator()(T x, T y) const {
      return max(x, y);
   }
};

struct add_op {
   __device__
   int operator()(int x, int y) const {
      return x + y;
   }
};

template <class T, class op_t>
__global__ void reduce_kernel(T *values, const size_t size, T ident, int k) {
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   op_t op;

   if (tid < k) {
      int idx = tid + k;
      values[tid] = op(values[tid], idx < size ? values[idx] : ident);
   }
}

template <class T, class op_t>
void reduce(T *result, const T * const d_array, const size_t size, T ident, int gridSize, int blockSize) {
   T *d_temp_array;
   checkCudaErrors(cudaMalloc(&d_temp_array, size * sizeof(T)));
   checkCudaErrors(cudaMemcpy(d_temp_array, d_array, size * sizeof(T), cudaMemcpyDeviceToDevice));

   int power_of_two = pow(2, ceil(log2f(size)));
   for (int k=2; k <= power_of_two; k <<= 1) {
      reduce_kernel<T, op_t><<<gridSize, blockSize>>>(d_temp_array, size, ident, power_of_two / k);
   }
   checkCudaErrors(cudaMemcpy(result, d_temp_array, sizeof(T), cudaMemcpyDeviceToHost));
}

template <class T, class op_t>
__global__ void reduce_block_kernel(T *values, T *out, int size) {
   int offset = blockIdx.x * blockDim.x;
   int tid = offset + threadIdx.x;
   op_t op;

   __shared__ T shm[1024];
   shm[threadIdx.x] = values[tid];
   __syncthreads();

   for (int p = pow(2, ceil(log2f(blockDim.x))); p >= 1; p >>= 1) {
      if (threadIdx.x < p >> 1) {
         int idx = threadIdx.x + (p >> 1);
         shm[threadIdx.x] = op(shm[threadIdx.x], shm[idx]);
      }
      __syncthreads();
   }
   if (threadIdx.x == 0) {
      out[blockIdx.x] = shm[threadIdx.x];
   }
}

template <class T, class op_t>
void reduce_block(T *result, const T * const d_array, size_t size, int gridSize, int blockSize) {
   T *d_in, *d_out;
   checkCudaErrors(cudaMalloc(&d_in, size * sizeof(T)));
   checkCudaErrors(cudaMemcpy(d_in, d_array, size * sizeof(T), cudaMemcpyDeviceToDevice));
   checkCudaErrors(cudaMalloc(&d_out, size * sizeof(T)));
   checkCudaErrors(cudaMemcpy(d_out, d_array, size * sizeof(T), cudaMemcpyDeviceToDevice));

   while (size > 1) {
      reduce_block_kernel<T, op_t><<<gridSize, blockSize>>>(d_in, d_out, size);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
      d_in = d_out;
      size = ceil((T) size / blockSize);
   }
   checkCudaErrors(cudaMemcpy(result, d_out, sizeof(T), cudaMemcpyDeviceToHost));
}

__global__ void init_hist(
   int *threadHists, int numHists, const float * const values, int num_vals, float min_val, float max_val, int numBins
) {
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   if (tid >= num_vals) {
      return;
   }
   for (int i = 0; i < numBins; ++i) {
      threadHists[i * numHists + tid] = 0;
   }

   int valsPerThread = ceil((float) num_vals / numHists);
   int start = min(tid * valsPerThread, num_vals);
   int end = min(start + valsPerThread, num_vals);
   float binSize = (float) numBins / (max_val - min_val);

   for (int i = start; i < end; ++i) {
      int bin = floor(binSize * (values[i] - min_val));
      bin = min(bin, numBins - 1);  // makes final bin inclusive of upper bound
      ++threadHists[bin * numHists + tid];
   }
}

void build_hist(
   int *hist, const float * const values, int num_vals, float min_val, float max_val, int numBins, int gridSize, int blockSize
) {
   int *d_threadHists;
   int numHists = min(num_vals, blockSize * gridSize);
   int totalHistSize = numHists * numBins * sizeof(int);
   checkCudaErrors(cudaMalloc(&d_threadHists, totalHistSize * sizeof(int)));
   checkCudaErrors(cudaMemset(d_threadHists, 0, totalHistSize * sizeof(int)));
   init_hist<<<gridSize, blockSize>>>(d_threadHists, numHists, values, num_vals, min_val, max_val, numBins);
   int *h_threadHists = (int *) malloc(totalHistSize * sizeof(int));
   checkCudaErrors(cudaMemcpy(h_threadHists, d_threadHists, totalHistSize * sizeof(int), cudaMemcpyDeviceToHost));

   int *h_bin = (int *) malloc(numHists * sizeof(int));
   int *d_hist, *d_bin;
   checkCudaErrors(cudaMalloc(&d_hist, sizeof(int)));
   checkCudaErrors(cudaMalloc(&d_bin, numHists * sizeof(int)));
   for (int i = 0; i < numBins; ++i) {
      reduce<int, add_op>(&hist[i], d_threadHists + i * numHists, numHists, 0, gridSize, blockSize);
   }
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

   size_t numPixels = numRows * numCols;
   int blockSize = 1024;
   int gridSize = ceil((float) numPixels / blockSize);

   reduce_block<float, max_op<float>>(&max_logLum, d_logLuminance, numPixels, gridSize, blockSize);
   reduce_block<float, min_op<float>>(&min_logLum, d_logLuminance, numPixels, gridSize, blockSize);

   int *h_hist = (int *) malloc(numBins * sizeof(int));
   build_hist(h_hist, d_logLuminance, numPixels, min_logLum, max_logLum, numBins, gridSize, blockSize);
}
