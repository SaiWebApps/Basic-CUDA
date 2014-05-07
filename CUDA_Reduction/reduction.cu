/******************************************************
 * CUDA Sum Reduction
 * By: Sairam Krishnan
 * Date: May 6, 2014
 * Compile command: nvcc -arch=sm_20 reduction.cu
 ******************************************************/

#include <cuda.h>
#include <stdio.h>

#define N 10
#define NTHRDS 4
#define NBLKS (((N) + (NTHRDS-1)) / (NTHRDS))

__global__ void sumReduction(int *input, int *output) {
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   int sum = 0, i;
   __shared__ int temp[NTHRDS];

   //Load the values into shared memory.
   temp[threadIdx.x] = input[index];
   //Wait for all threads in the current block to finish loading values.
   __syncthreads();
 
   //Offload the reduction work for this block to thread 0.
   if (threadIdx.x != 0)
      return;
   for (i = 0; i<blockDim.x; i++) { 
      if (index+i >= N) 
	break;
      sum += temp[i];
   }

   //Atomic add to prevent inteference from threads outside this block
   atomicAdd(output, sum);
}

int main() {
   int input[N], output, i;
   int *devInput, *devOutput;

   for (i = 0; i<N; i++)
	input[i] = i+1;
   
   cudaMalloc(&devInput, N*sizeof(int));
   cudaMalloc(&devOutput, sizeof(int));
   cudaMemcpy(devInput, input, N*sizeof(int), cudaMemcpyHostToDevice);
   cudaMemset(devOutput, 0, sizeof(int));

   sumReduction <<<NBLKS, NTHRDS>>>(devInput, devOutput);

   cudaMemcpy(&output, devOutput, sizeof(int), cudaMemcpyDeviceToHost);
   
   printf("%d\n", output);

   cudaFree(devOutput);
   cudaFree(devInput);
   return 0;
}
