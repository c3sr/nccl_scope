#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <nccl.h> 

#include <cuda_runtime.h>

#include "scope/init/init.hpp"

#include "SingleProcess/args.hpp"

#define NAME "NCCL/function/ALLREDUCEOrig"

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

static void NCCL_function_ALLREDUCEOrig(benchmark::State &state) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));
  const int gpu0 = 0;
  const int gpu1 = 1;
  const int gpu2 = 2;
  const int gpu3 = 3;

  if (PRINT_IF_ERROR(utils::cuda_reset_device(gpu0))) {
    state.SkipWithError(NAME " failed to reset CUDA device");
    return;
  }
  if (PRINT_IF_ERROR(utils::cuda_reset_device(gpu1))) {
    state.SkipWithError(NAME " failed to reset CUDA device");
    return;
  }
  if (PRINT_IF_ERROR(utils::cuda_reset_device(gpu2))) {
    state.SkipWithError(NAME " failed to reset CUDA device");
    return;
  }
  if (PRINT_IF_ERROR(utils::cuda_reset_device(gpu3))) {
    state.SkipWithError(NAME " failed to reset CUDA device");
    return;
  }
  ncclComm_t comms[4];
 
  //4 devices
  int nDev = 4;
  int size = 32 * 1024 *1024;
  int devs[4] = {gpu0,gpu1, gpu2, gpu3 };


  // figure out how to shorten this part

  cudaEvent_t start1, start2, start3, start4, stop1, stop2, stop3, stop4;
  std::vector<cudaEvent_t> starts = {start1, start2, start3, start4};
  std::vector<cudaEvent_t> stops = {stop1, stop2, stop3, stop4};
  CUDACHECK(  cudaEventCreate(&starts[0]));
  CUDACHECK(  cudaEventCreate(&starts[1]));
  CUDACHECK(  cudaEventCreate(&starts[2]));
  CUDACHECK(  cudaEventCreate(&starts[3]));

  CUDACHECK(  cudaEventCreate(&stops[0]));
  CUDACHECK(  cudaEventCreate(&stops[1]));
  CUDACHECK(  cudaEventCreate(&stops[2]));
  CUDACHECK(  cudaEventCreate(&stops[3]));

  

   //allocate and inialize device buffer 
//made memset all 0 instead of  0 first and 1 second
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
//  cudaEvent_t* starts = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*nDev);
//  cudaEvent_t* stops = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*nDev);  
  
for (int i = 0; i < nDev; ++i) {
    
    if(PRINT_IF_ERROR(cudaSetDevice(devs[i]))){
      state.SkipWithError(NAME "failed to set device");
      return;
    }
    if(PRINT_IF_ERROR(cudaMalloc(sendbuff + i, size * sizeof(float) ))){
      state.SkipWithError(NAME "failed to do cudaMalloc  sendbuff");
      return;
    }
    if(PRINT_IF_ERROR(cudaMalloc(recvbuff + i, size * sizeof(float) ))){
      state.SkipWithError(NAME "failed to do cudaMalloc recvbuff");
      return;
    }

    if(PRINT_IF_ERROR(cudaMemset(sendbuff[i], 1, size * sizeof(float) ))){
      state.SkipWithError(NAME "failed to do  cudaMemset sendbuff");
      return;
    }
    if(PRINT_IF_ERROR(cudaMemset(recvbuff[i], 0, size * sizeof(float) ))){
      state.SkipWithError(NAME "failed to do cudaMemset recvbuff");
      return;
    }
    if(PRINT_IF_ERROR(cudaStreamCreate(streams+i))){
      state.SkipWithError(NAME "failed to create stream");
      return;
    }

/*
    if(PRINT_IF_ERROR(cudaEventCreate(starts+i))){
      state.SkipWithError(NAME "failed to create start");
      return;
    }

    if(PRINT_IF_ERROR(cudaEventCreate(stops+i))){
      state.SkipWithError(NAME "failed to create stop");
      return;
    }
*/
  }

  ncclCommInitAll(comms, nDev , devs);
/*
assert(starts.size() == stops.size());
assert(streams.size() == starts.size());
assert(sendbuff.size() == recvbuff.size());
assert(streams.size() == sendbuff.size());
*/
//benchmark loop
//for(auto _ : state){
NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nDev; ++i) {
      auto start = starts[i];
      auto stop = stops[i];
      auto stream = streams[i];
      auto comm = comms[i];
      if(PRINT_IF_ERROR(cudaEventRecord(start, stream ))) {
        state.SkipWithError(NAME " failed to record start event");
        return;
      }
      if(PRINT_IF_ERROR(ncclAllReduce(sendbuff[i], recvbuff[i], size, ncclFloat, ncclSum, comm, stream))){
           state.SkipWithError(NAME "failed to do ncclAllReduce");
      }
      if(PRINT_IF_ERROR(cudaEventRecord(stop, stream))) {
        state.SkipWithError(NAME " failed to record stop event");
        return;
      }

    }
NCCLCHECK(ncclGroupEnd());
    //synchronizing on CUDA streams to wait for completion of NCCL operation
   for (int i = 0; i < nDev; ++i) {
     if(PRINT_IF_ERROR(cudaSetDevice(devs[i]))){
       state.SkipWithError(NAME "failed to set device");
       return;
     }
     if(PRINT_IF_ERROR(cudaStreamSynchronize(streams[i]))){
       state.SkipWithError(NAME "failed to synchronize");
       return;
     }
   }

    // Find the longest time between any start and stop
    float maxMillis = 0;
    for (const auto start : starts) {
      for ( const auto stop : stops) {
        float millis;

        if (PRINT_IF_ERROR(cudaEventElapsedTime(&millis, start, stop))) {
          state.SkipWithError(NAME " failed to synchronize");
          return;
        }

        maxMillis = std::max(millis, maxMillis);
      }
    }


    state.SetIterationTime(maxMillis / 1000);

//}

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes) * 2);
  state.counters.insert({{"bytes", bytes}});

  //free device buffers
  for (int i = 0; i < nDev; ++i) {
    cudaSetDevice(devs[i]);
    cudaFree(sendbuff[i]);
    cudaFree(recvbuff[i]);
  }  

   //destory communicator objects and free memory
   for (int i=0; i<4; i++){
     ncclCommDestroy(comms[i]);
   }
}
BENCHMARK(NCCL_function_ALLREDUCEOrig)->Apply(ArgsCountGpuGpuGpuGpu)->UseManualTime();

