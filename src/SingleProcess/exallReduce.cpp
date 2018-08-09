#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <nccl.h>

#include <cuda_runtime.h>

#include "scope/init/init.hpp"

#include "SingleProcess/args.hpp"

#define NAME "NCCL/function/EXALLREDUCE"
//time problems only 

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)
#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


static void NCCL_function_EXALLREDUCE(benchmark::State &state) {
  ncclComm_t comms[4];

  //managing 4 devices
  int nDev = 4;
  int size = 32*1024*1024;
  int devs[4] = { 0, 1, 2, 3 };
  
  //allocating and initializing device buffers
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
  cudaEvent_t* starts = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*nDev);
  cudaEvent_t* stops = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*nDev);
  //cudaEvent_t* zeros = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*nDev);
  
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s+i));
    CUDACHECK(cudaEventCreate(starts+i));
    CUDACHECK(cudaEventCreate(stops+i));
   // CUDACHECK(cudaEventCreate(zeros+i));
  }
 //nitializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));
for(auto _ : state){
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i){
    CUDACHECK(cudaEventRecord(starts[i], s[i]));
    NCCLCHECK(ncclAllReduce(sendbuff[i], recvbuff[i], size, ncclFloat, ncclSum,
        comms[i], s[i]));
    CUDACHECK(cudaEventRecord(stops[i], s[i]));
    //CUDACHECK(cudaEventRecord(zeros[i], s[i]));
 }
  NCCLCHECK(ncclGroupEnd());

  //synchronize
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaEventSynchronize(stops[i]));
    CUDACHECK(cudaStreamSynchronize(s[i])); 
  }
/*
//largest time diff between starts and stops
  float maxMillis = 0; 
    for (int i = 0; i < nDev; ++i ) {
     for (int j =0; j < nDev; ++j) {
        //use eventelapsed against a zero
        float millis;
	float start;
	float stop;
        CUDACHECK(cudaSetDevice(i));
 	CUDACHECK(cudaEventElapsedTime(&millis, starts[i] , stops[j])); 
        //CUDACHECK(cudaSetDevice(j));
        //CUDACHEKC(cudaSetDevice(j));
        //CUDACHECK(cudaEventTimeElapsed(&stop, 0, stops[j]));
       maxMillis = std::max(millis, maxMillis);

    }

  }
   state.SetIterationTime(maxMillis / 1000);
*/
}
//state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes) * 2);
//state.counters.insert({{"bytes", bytes}});

  //free device buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
  }

  //destroy comms
  for(int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);

}
BENCHMARK(NCCL_function_EXALLREDUCE)->Apply(ArgsCountGpuGpuGpuGpu);

