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

#define OR_SKIP(stmt, msg) \
  if (PRINT_IF_ERROR(stmt)) { \
    state.SkipWithError(msg); \
    return; \
  }

#define NCCL_SKIP(cmd,msg) {                         \
  ncclResult_t e = cmd;                              \
  if( e != ncclSuccess ) {                          \
    state.SkipWithError(msg);   \
    return;                             \
  }                                                 \
}


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
    OR_SKIP(cudaSetDevice(i), NAME " failed to set device");
    OR_SKIP(cudaMalloc(sendbuff + i, bytes * sizeof(float)), NAME " failed to perform cudaMalloc");
    OR_SKIP(cudaMalloc(recvbuff + i, bytes * sizeof(float) * nDev), NAME " failed to perform cudaMalloc");
    OR_SKIP(cudaMemset(sendbuff[i], 1, bytes * sizeof(float)  ), NAME " failed to perform cudaMemset");
    OR_SKIP(cudaMemset(recvbuff[i], 0, bytes * sizeof(float) * nDev), NAME " failed to perform cudaMemset");
    OR_SKIP(cudaStreamCreate(s+i), NAME " failed to create stream");
    OR_SKIP(cudaEventCreate(starts+i), NAME " failed to create event");
    OR_SKIP(cudaEventCreate(stops+i), NAME " failed to create event");
  }

  cudaEvent_t start, stop;
  OR_SKIP(cudaEventCreate(&start), NAME " failed to create event");
  OR_SKIP(cudaEventCreate(&stop), NAME " failed to create event");

  //initializing NCCL
  NCCL_SKIP(ncclCommInitAll(comms, nDev, devs), NAME " failed to initialize comm");
  for(auto _ : state){

  OR_SKIP(cudaEventRecord(start, NULL), NAME " failed to record event");

  NCCL_SKIP(ncclGroupStart(), NAME " failed to start group");
  for (int i = 0; i < nDev; ++i){
    OR_SKIP(cudaEventRecord(starts[i], s[i]), NAME " failed to record event");
    NCCL_SKIP(ncclAllGather(sendbuff[i], recvbuff[i], bytes, ncclFloat,
        comms[i], s[i]), NAME " failed to perform allGather");
    OR_SKIP(cudaEventRecord(stops[i], s[i]), NAME " failed to record event");
  }
  NCCL_SKIP(ncclGroupEnd(), NAME " failed to stop group");
  OR_SKIP(cudaEventRecord(stop, NULL), NAME " failed to record event");

  //synchronize
  for (int i = 0; i < nDev; ++i) {
    OR_SKIP(cudaStreamSynchronize(s[i]),NAME " failed to synchronize streams" ); 
  }
  OR_SKIP(cudaEventSynchronize(stop), NAME " failed to synchronize events");

  //timing
  state.PauseTiming();
  float msecTotal = 0.0f;
  OR_SKIP(cudaEventElapsedTime(&msecTotal, start, stop), NAME " failed to compute elapse time" );

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
    OR_SKIP(cudaEventElapsedTime(&device[i], starts[i] , stops[i]), NAME "failed to compare times");
    total += device[i];	
  }

  state.counters["d0"] = device[0];
  state.counters["d1"] = device[1];
  state.counters["d2"] = device[2];
  state.counters["d3"] = device[3];
  state.counters["avg"]= total/nDev;

  //free device buffers
  for (int i = 0; i < nDev; ++i) {
    OR_SKIP(cudaSetDevice(i), NAME " failed to set device");
    OR_SKIP(cudaFree(sendbuff[i]), NAME " failed to free sendbuff");
    OR_SKIP(cudaFree(recvbuff[i]), NAME " failed to free recvbuff");
  }

  //destroy comms
  for(int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);

}
BENCHMARK(NCCL_ops_allGather)->Apply(ArgsCountGpuGpuGpuGpu)->UseManualTime();

