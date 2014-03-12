#include <stdio.h>

#define NUM_ELEMS 4

__global__ void multiply(int *a, int *b, int *c) {
   c[blockIdx.x] =  a[blockIdx.x] * b[blockIdx.x];
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
  //Specify NUM_ELEMS thread blocks, each with 1 thread.
  multiply <<< NUM_ELEMS,1 >>>(device1, device2, device3);

  //Get result from device to host.
  cudaMemcpy(&output, device3, numBytes, cudaMemcpyDeviceToHost);

  //Print out values.
  for (i = 0; i < NUM_ELEMS; i++) {
    printf("%d * %d = %d\n", host1[i], host2[i], output[i]);
  }

  //Free all variables.
  cudaFree(device1);
  cudaFree(device2);
  cudaFree(device3);

  return 0;
}
