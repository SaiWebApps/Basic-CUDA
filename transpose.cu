#include <stdio.h>

#define DIM 4
#define NUM_ELEMS DIM*DIM

__global__ void transpose(int *a, int *b) {
   int row = blockIdx.x * DIM/2 + threadIdx.x;
   int col = blockIdx.y * DIM/2 + threadIdx.y;
   int newIndex = row * DIM + col;
   int oldIndex = col * DIM + row;
   b[newIndex] = a[oldIndex];
}

int main() {
  //device memory
  int *device1, *device2;
  //host memory
  int host[NUM_ELEMS];
  int output[NUM_ELEMS];
  size_t numBytes = NUM_ELEMS * sizeof(int);
  int i = 0; //loop counter

  //Load host1 and host2 with values.
  for (i = 0; i < NUM_ELEMS; i++) {
     host[i] = i+1;
  }
  
  //Allocate memory for device vars.
  cudaMalloc((void **)&device1, numBytes);
  cudaMalloc((void **)&device2, numBytes);

  //Transfer values from host to device.
  cudaMemcpy(device1, &host, numBytes, cudaMemcpyHostToDevice);

  //Launch transpose kernel on GPU with given parameters.
  dim3 grid(DIM/2, DIM/2); //# of thread blocks
  dim3 block(DIM/2, DIM/2); //# of threads per thread block
  transpose<<<grid,block>>>(device1, device2);

  //Get result from device to host.
  cudaMemcpy(&output, device2, numBytes, cudaMemcpyDeviceToHost);

  //Print out values.
  printf("[");
  for (i = 0; i < NUM_ELEMS; i++) {
    printf("%d ", output[i]);
  }
  printf("]\n");

  //Free all variables.
  cudaFree(device1);
  cudaFree(device2);

  return 0;
}
