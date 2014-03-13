#include <stdio.h>
#include <time.h>

#define NUM_ELEMS 16
#define BLOCK_SIZE 4

__global__ void multiply(int *a, int *b, int *c) {
   int index = blockIdx.x * blockDim.x + threadIdx.x;
   __shared__ int shared_mem[BLOCK_SIZE];   

   //Copy a and b into shared memory.
   shared_mem[threadIdx.x] = a[index] * b[index];
   //Transfer result from shared memory into output array c.
   c[index] = shared_mem[threadIdx.x];
}

int main() {
  //device memory
  int *device1, *device2, *device3;
  //host memory
  int host1[NUM_ELEMS];
  int host2[NUM_ELEMS];  
  int output[NUM_ELEMS];
  size_t numBytes = NUM_ELEMS * sizeof(int);
  int i = 0; //loop counter
  clock_t cpu_time = clock(); //Start CPU clock.
  float time = 0.0f;
  cudaEvent_t start, stop;

  //Start GPU clock.
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  //Load host1 and host2 with values.
  for (i = 0; i < NUM_ELEMS; i++) {
     host1[i] = i+1;
     host2[i] = i+5;
  }
  
  //Allocate memory for device vars.
  cudaMalloc((void **)&device1, numBytes);
  cudaMalloc((void **)&device2, numBytes);
  cudaMalloc((void **)&device3, numBytes);

  //Transfer values from host to device.
  cudaMemcpy(device1, &host1, numBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device2, &host2, numBytes, cudaMemcpyHostToDevice);

  //Launch multiply kernel on GPU with given parameters.
  //Specify NUM_ELEMS thread blocks, each with BLOCK_SIZE threads.
  multiply <<<NUM_ELEMS/BLOCK_SIZE,BLOCK_SIZE>>>(device1, device2, device3);

  //Get result from device to host.
  cudaMemcpy(&output, device3, numBytes, cudaMemcpyDeviceToHost);

  //Stop GPU clock - determine how long GPU kernel took to run.
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  //Print out values.
  for (i = 0; i < NUM_ELEMS; i++) {
    printf("%d * %d = %d\n", host1[i], host2[i], output[i]);
  }

  //Free all variables.
  cudaFree(device1);
  cudaFree(device2);
  cudaFree(device3);

  //Calculate total runtime.
  cpu_time = clock() - cpu_time; //CPU time
  time += ((double)cpu_time)/CLOCKS_PER_SEC; //Add CPU time to GPU time.
  printf("%f\n", time); //print out total runtime
  return 0;
}
