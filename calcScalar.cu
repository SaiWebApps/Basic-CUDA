#include <stdio.h>

__global__ void add(int *a, int *b, int *c) {
   *c = (*a) + (*b);
}

__global__ void multiply(int *a, int *b, int *c) {
   *c = (*a) * (*b);
}

__global__ void subtract(int *a, int *b, int *c) {
   *c = (*a) - (*b);
}

__global__ void divide(int *a, int *b, int *c) {
   *c = (*a) / (*b);
}

int main() {
  int host1, host2, output; //host variables
  char op; //host variable
  int *device1, *device2, *device3; //device memory

  //Allocate memory for device vars.
  cudaMalloc((void **)&device1, sizeof(int));
  cudaMalloc((void **)&device2, sizeof(int));
  cudaMalloc((void **)&device3, sizeof(int));

  //Read 2 integers. Store values in host variables.
  printf("Enter two integers: ");
  fscanf(stdin, "%d %d %c", &host1, &host2, &op);
  
  //Transfer values from host to device.
  cudaMemcpy(device1, &host1, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device2, &host2, sizeof(int), cudaMemcpyHostToDevice);

  //Launch add kernel on GPU with given parameters.
  switch(op) {
     case '+': add <<< 1,1 >>>(device1, device2, device3); break;
     case '*': multiply <<< 1,1 >>>(device1, device2, device3); break;
     case '-': subtract <<< 1,1 >>>(device1, device2, device3); break;
     case '/': divide <<< 1,1 >>>(device1, device2, device3); break;
  }
  //Get result from device to host.
  cudaMemcpy(&output, device3, sizeof(int), cudaMemcpyDeviceToHost);

  //Print result.
  printf("%d %c %d = %d\n", host1, op, host2, output);

  //Free all variables.
  cudaFree(device1);
  cudaFree(device2);
  cudaFree(device3);

  return 0;
}
