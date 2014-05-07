/******************************************************
 * CUDA Sum Reduction
 * By: Sairam Krishnan
 * Date: May 6, 2014
 * Compile command: nvcc -arch=sm_20 reduction.cu
 ******************************************************/

#include <cuda.h>
#include <stdio.h>

#define N 100
#define NTHRDS 8
#define NBLKS (((N) + (NTHRDS-1)) / (NTHRDS))

int getNearestPowerOfTwo(int n) {
    int count = 1;
    while (n > 1) {
      n >>= 1;
      count <<= 1;
    }
    return count << 1;
}

__global__ void sumReduction(int *input, int *output) {
   int i, tidx = threadIdx.x, index = blockIdx.x*blockDim.x + tidx;
   extern __shared__ int temp[];
  
   if (index >= N) {
     return;
   }
 
   temp[tidx] = input[index];
   __syncthreads();

   for (i=blockDim.x/2; i>0; i>>=1) {
       if (tidx < i && index+i < N) {
  	  temp[tidx] += temp[tidx + i];
       }
       __syncthreads();
   }

   if (tidx == 0) {
      output[blockIdx.x] = temp[0];
   }
}

int main() {
   int nb = getNearestPowerOfTwo(NBLKS), i, sum;
   int input[N], output[nb];
   int *devInput, *devOutput1, *devOutput2;
   
   //Sizes
   int INT_SIZE = sizeof(int);
   int N_SIZE = N * INT_SIZE;
   int NBLKS_SIZE = NBLKS * INT_SIZE;
   int NTHRDS_SIZE = NTHRDS * INT_SIZE;
   int NB_SIZE = nb * INT_SIZE;

   //Load host input array with values {1....N}. Clear output array for stage 1.
   for (i = 0; i<N; i++)
	input[i] = i+1;
   memset(output, 0, NB_SIZE); 

   //Allocate memory for device pointers.
   cudaMalloc(&devInput, N_SIZE);
   cudaMalloc(&devOutput1, NB_SIZE);
   cudaMalloc(&devOutput2, INT_SIZE);

   //Load input array into input device pointer.
   //Clear stage 1 output device pointer.
   cudaMemcpy(devInput, input, N_SIZE, cudaMemcpyHostToDevice);
   cudaMemcpy(devOutput1, output, NB_SIZE, cudaMemcpyHostToDevice);
 
   //Execute stage 1 reduction. Partial sums will be stored in devOutput1.
   //Copy partial sums from devOutput1 to output for debugging purposes.
   sumReduction <<<NBLKS, NTHRDS, NTHRDS_SIZE>>>(devInput, devOutput1);
   cudaMemcpy(output, devOutput1, NBLKS_SIZE, cudaMemcpyDeviceToHost);

   //Stage 2 reduction - add up the partial sums and store final result in sum
   sumReduction <<<1, nb, NB_SIZE>>>(devOutput1, devOutput2);
   cudaMemcpy(&sum, devOutput2, INT_SIZE, cudaMemcpyDeviceToHost);

   //Print out partial sums and final sum.
   for (i = 0; i<nb; i++) {
     printf("%d ", output[i]);
   }
   printf("\n%d\n", sum);

   //Free allocated device memory.
   cudaFree(devOutput2);
   cudaFree(devOutput1);
   cudaFree(devInput);
   return 0;
}
