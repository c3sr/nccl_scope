#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <nccl.h>
#include <cuda_runtime.h>

#include "scope/init/init.hpp"
#include "scope/utils/utils.hpp"
#include "scope/init/flags.hpp"
#include "init/flags.hpp"
#include "SingleProcess/args.hpp"

#define NAME "NCCL/ops/allGather"

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


static void NCCL_ops_allGather(benchmark::State &state) {
  const int nDev = FLAG(ngpu);
  ncclComm_t comms[nDev];

  //managing ngpu devices
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));
  int devs[nDev];
  for(int i = 0; i < nDev; ++i){
     devs[i]= FLAG(cuda_device_ids)[i];
  }

  //allocating and initializing device buffers
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
  cudaEvent_t* starts = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*nDev);
  cudaEvent_t* stops = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*nDev);

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(sendbuff + i, bytes * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, bytes * sizeof(float) * nDev));
    CUDACHECK(cudaMemset(sendbuff[i], 1, bytes * sizeof(float)  ));
    CUDACHECK(cudaMemset(recvbuff[i], 0, bytes * sizeof(float) * nDev));
    CUDACHECK(cudaStreamCreate(s+i));
    CUDACHECK(cudaEventCreate(starts+i));
    CUDACHECK(cudaEventCreate(stops+i));
  }

  cudaEvent_t start, stop;
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&stop));

  //initializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));
  for(auto _ : state){

  CUDACHECK(cudaEventRecord(start, NULL));

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i){
    CUDACHECK(cudaEventRecord(starts[i], s[i]));
    NCCLCHECK(ncclAllGather(sendbuff[i], recvbuff[i], bytes, ncclFloat,
        comms[i], s[i]));
    CUDACHECK(cudaEventRecord(stops[i], s[i]));
  }
  NCCLCHECK(ncclGroupEnd());
  CUDACHECK(cudaEventRecord(stop, NULL));

  //synchronize
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaStreamSynchronize(s[i])); 
  }
  CUDACHECK(cudaEventSynchronize(stop));

  //timing
  state.PauseTiming();
  float msecTotal = 0.0f;
  CUDACHECK(cudaEventElapsedTime(&msecTotal, start, stop));

  state.SetIterationTime(msecTotal/ 1000);
  state.ResumeTiming();

  }

  state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(bytes));
  state.counters.insert({{"bytes", bytes}});

  //device time comparisons
  float d0, d1, d2, d3;
  float total = 0.0f;
  std::vector<float> device = {d0, d1, d2, d3};
  for (int i = 0; i < nDev; ++i ) {
    CUDACHECK(cudaEventElapsedTime(&device[i], starts[i] , stops[i]));
    total += device[i];	
  }

  state.counters["d0"] = device[0];
  state.counters["d1"] = device[1];
  state.counters["d2"] = device[2];
  state.counters["d3"] = device[3];
  state.counters["avg"]= total/nDev;

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
BENCHMARK(NCCL_ops_allGather)->Apply(ArgsCountGpuGpuGpuGpu)->UseManualTime();

