#include <cuda.h>
#include <stdio.h>
#include <time.h>

#define N 10

typedef struct {
   float x[N];
   float y[N];
   float z[N];
} coord;

__global__
void compute_dist2(float *x, float *y, float *z, float *result) {
   int index = threadIdx.x;
   float deltaX = x[index+1] - x[index]; 
   float deltaY = y[index+1] - y[index];
   float deltaZ = z[index+1] - z[index];
   result[index] = deltaX*deltaX + deltaY*deltaY + deltaZ*deltaZ;
}

void loadRandomCoords(coord *c) {
  int i;
  srand(time(NULL));

  for (i = 0; i<N; i++) {
     c->x[i] = rand();
     c->y[i] = rand();
     c->z[i] = rand();
  }
}

void loadTestCoords(coord *c) {
  int i;
  for (i = 0; i<N; i++) {
     c->x[i] = i+1;
     c->y[i] = i+2;
     c->z[i] = i+3;
  }
}

void printArray(float *f, int len) {
  int i;
  for (i = 0; i<len; i++) {
    printf("%f ", f[i]);
  }
  printf("\n");
}

int main() {
  coord c;
  float result[N-1];
  float *dev_x, *dev_y, *dev_z, *dev_result;
  dim3 grid(1,1), block(N-1,1);

  //Load coordinates into host arrays.
  loadRandomCoords(&c);

  //Allocate memory for device pointers.
  cudaMalloc(&dev_x, N*sizeof(float));
  cudaMalloc(&dev_y, N*sizeof(float));
  cudaMalloc(&dev_z, N*sizeof(float));
  cudaMalloc(&dev_result, (N-1)*sizeof(float));

  //Transfer coordinates from host to device.
  cudaMemcpy(dev_x, c.x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_y, c.y, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_z, c.z, N*sizeof(float), cudaMemcpyHostToDevice);
  
  //Call kernel to compute euclidean distance b/w adjacent points squared.
  //Be sure to only use device pointers since device can't access host mem
  //and vice versa.
  compute_dist2 <<<grid,block>>>(dev_x, dev_y, dev_z, dev_result);

  //Transfer results from device memory to host memory.
  cudaMemcpy(result, dev_result, (N-1)*sizeof(float), cudaMemcpyDeviceToHost);

  //Free device memory.
  cudaFree(dev_x);
  cudaFree(dev_y);
  cudaFree(dev_z);
  cudaFree(dev_result);

  printArray(result, N-1);
  return 0;
}
