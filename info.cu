#include <stdio.h>

int main() {
  int nDevices, i;
  cudaDeviceProp prop;
  cudaGetDeviceCount(&nDevices);
  
  for(i = 0; i<nDevices; i++) {
     cudaGetDeviceProperties(&prop, i);
     printf("Name: %s, Major: %d, Minor: %d\n", prop.name, prop.major, prop.minor);
     printf("Maximum # of Threads Per Block = %d\n", prop.maxThreadsPerBlock);
     printf("Maximum # of Threads Per Multiprocessor = %d\n", prop.maxThreadsPerMultiProcessor);
     printf("# of Threads Per Warp = %d\n", prop.warpSize);
     printf("# of Registers Per Block = %d\n", prop.regsPerBlock);
     printf("Size of L2 Cache in bytes = %d\n", prop.l2CacheSize);
     printf("Total Global Memory = %d\n", prop.totalGlobalMem);
     printf("Total Shared Memory Per Block = %d\n", prop.sharedMemPerBlock);
     printf("Total # of Multiprocessors = %d\n", prop.multiProcessorCount);
     printf("Clock rate = %d\n", prop.clockRate);
     printf("\n");
  }
}
