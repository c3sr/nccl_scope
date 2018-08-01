#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <nccl.h> 

#include <cuda_runtime.h>

#include "scope/init/init.hpp"

#include "SingleProcess/args.hpp"

#define NAME "NCCL/function/REDUCESCATTER"
//combines data from all sender and delvers to all participants
//single process that manages four devices/gpus 
static void NCCL_function_REDUCESCATTER(benchmark::State &state) {

  if (!has_cuda) {
    state.SkipWithError(NAME " no CUDA device found");
    return;
  }

  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));
  const int gpu0 = state.range(1);
  const int gpu1 = state.range(2);
  const int gpu2 = state.range(3);
  const int gpu3 = state.range(4);

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
  //create a communicator object
  //single process: communicator for 4 devices therefore, there  are 4 communicator objects:
  ncclComm_t comms[4];
 
  //4 devices
  int nDev = 4;
  int size = bytes;
  int devs[4] = {gpu0,gpu1, gpu2, gpu3 };


  // figure out how to shorten this part
  cudaEvent_t start1, start2, start3, start4, stop1, stop2, stop3, stop4;
  std::vector<cudaEvent_t> starts = {start1, start2, start3, start4};
  std::vector<cudaEvent_t> stops = {stop1, stop2, stop3, stop4};
  cudaEventCreate(&starts[0]);
  cudaEventCreate(&starts[1]);
  cudaEventCreate(&starts[2]);
  cudaEventCreate(&starts[3]);
  cudaEventCreate(&stops[0]);
  cudaEventCreate(&stops[1]);
  cudaEventCreate(&stops[2]);
  cudaEventCreate(&stops[3]);


  //allocate and inialize device buffer
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
 //added these two which  should create start adn stop evetns
 // cudaEvent_t* starts = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*nDev);
 // cudaEvent_t* stops = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*nDev);  
  for (int i = 0; i < nDev; ++i) {
    if(PRINT_IF_ERROR(cudaSetDevice(devs[i]))){
      state.SkipWithError(NAME "failed to set device");
      return;
    }
    if(PRINT_IF_ERROR(cudaMalloc(sendbuff + i, size * sizeof(float)))){
      state.SkipWithError(NAME "failed to do cudaMalloc sendbuff");
      return;
    }
    if(PRINT_IF_ERROR(cudaMalloc(recvbuff + i, size * sizeof(float)))){
      state.SkipWithError(NAME "failed to do cudaMalloc  recvbuff");
      return;
    }
    if(PRINT_IF_ERROR(cudaMemset(sendbuff[i], 1, size * sizeof(float)))){
      state.SkipWithError(NAME "failed to do  cudaMemset sendbuff");
      return;
    }
    if(PRINT_IF_ERROR(cudaMemset(recvbuff[i], 0, size * sizeof(float)))){
      state.SkipWithError(NAME "failed to do cudaMemset recvbuff");
      return;
    }
    if(PRINT_IF_ERROR(cudaStreamCreate(s+i))){
      state.SkipWithError(NAME "failed to create stream");
      return;
    }
    //added by me
/*
    if(PRINT_IF_ERROR(cudaEventCreate(starts+1))){
      state.SkipWithError(NAME "failed to create start");
      return;
    }
    if(PRINT_IF_ERROR(cudaEventCreate(stops+1))){
      state.SkipWithError(NAME "failed to create stop");
      return;
    }
*/
  }

  ncclCommInitAll(comms, 4 , devs);

  //assert that send buff == reveive bugg
  //assert number of streams etc. 
  // do all reduce here
  for (auto _ : state) {
    // Start copy
    for (int i = 0; i < nDev; ++i) {
      auto start = starts[i];
      auto stop = stops[i];
      auto stream = s[i];
      if(PRINT_IF_ERROR(cudaEventRecord(start, stream))) {
        state.SkipWithError(NAME " failed to record start event");
        return;
      }
      //multiple devices per thread
 ncclGroupStart();
        if(PRINT_IF_ERROR(ncclReduceScatter((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum, comms[i], stream))){
           state.SkipWithError(NAME "failed to do ncclAllReduce");
        }
 ncclGroupEnd();

      if(PRINT_IF_ERROR(cudaEventRecord(stop, stream))) {
        state.SkipWithError(NAME " failed to record stop event");
        return;
      }
    }
    
    //synchronizing on CUDA streams to wait for completion of NCCL operation
   for (int i = 0; i < nDev; ++i) {
     if(PRINT_IF_ERROR(cudaSetDevice(devs[i]))){
       state.SkipWithError(NAME "failed to set device");
       return;
      }
     if(PRINT_IF_ERROR(cudaStreamSynchronize(s[i]))){
       state.SkipWithError(NAME "faiel to synchronize");
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
  }
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
BENCHMARK(NCCL_function_REDUCESCATTER)->Apply(ArgsCountGpuGpuGpuGpu)->UseManualTime();

