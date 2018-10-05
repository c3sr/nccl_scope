#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <nccl.h>
#include <cuda_runtime.h>

#include "scope/utils/utils.hpp"
#include "scope/init/flags.hpp"
#include "scope/init/init.hpp"
#include "init/flags.hpp"
#include "SingleProcess/args.hpp"

#define NAME "NCCL/ops/reduce"

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


auto NCCL_ops_reduce = [](benchmark::State &state) {
  const int nDev = FLAG(ngpu);
  std::vector<ncclComm_t> communicator(nDev);
  ncclComm_t* comms = &communicator[0];

  //managing 4 devices
  const auto bytes = 1ULL << static_cast<size_t>(state.range(0));
  std::vector<int> devices(nDev);
  int* devs = &devices[0];

  for(int i = 0; i <nDev; ++i){
     devs[i]=FLAG(cuda_device_ids)[i];
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
    OR_SKIP(cudaMalloc(recvbuff + i, bytes * sizeof(float)), NAME " failed to perform cudaMalloc");
    OR_SKIP(cudaMemset(sendbuff[i], 1, bytes * sizeof(float)), NAME " failed to perform cudaMemset");
    OR_SKIP(cudaMemset(recvbuff[i], 0, bytes * sizeof(float)), NAME " failed to perform cudaMemset");
    OR_SKIP(cudaStreamCreate(s+i), NAME " failed to create streams");
    OR_SKIP(cudaEventCreate(starts+i), NAME " failed to create events");
    OR_SKIP(cudaEventCreate(stops+i), NAME " failed to create events");
  }

  cudaEvent_t start, stop;
  OR_SKIP(cudaEventCreate(&start), NAME " failed to create event");
  OR_SKIP(cudaEventCreate(&stop), NAME " failed to create event");

  //initializing NCCL
  NCCL_SKIP(ncclCommInitAll(comms, nDev, devs), NAME " failed to initialize comm");
  for(auto _ : state){

  OR_SKIP(cudaEventRecord(start, NULL), NAME " failed to record start event");

  NCCL_SKIP(ncclGroupStart(), NAME " failed to start group");
  for (int i = 0; i < nDev; ++i){
    OR_SKIP(cudaEventRecord(starts[i], s[i]), NAME " failed to record start event");
    NCCL_SKIP(ncclReduce(sendbuff[i], recvbuff[i], bytes, ncclFloat, ncclSum,0,
        comms[i], s[i]), NAME " failed to perform allReduce");
    OR_SKIP(cudaEventRecord(stops[i], s[i]), NAME " failed to record stop event");
  }
  NCCL_SKIP(ncclGroupEnd(), NAME " failed to stop group");
  OR_SKIP(cudaEventRecord(stop, NULL), NAME " failed to record stop event");

  //synchronize
  for (int i = 0; i < nDev; ++i) {
    OR_SKIP(cudaStreamSynchronize(s[i]), NAME " failed to synchronize streams"); 
  }
  OR_SKIP(cudaEventSynchronize(stop), NAME " failed to synchronize events");

  //timing
  state.PauseTiming();
  float msecTotal = 0.0f;
  OR_SKIP(cudaEventElapsedTime(&msecTotal, start, stop), NAME " failed to compute elapsed time");

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
     OR_SKIP(cudaEventElapsedTime(&device[i], starts[i] , stops[i]), NAME " failed to compare times");
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

};

static void registerer() {
  std::string name;
 // for (auto cuda_id : unique_cuda_device_ids()) {
   // for (auto numa_id : unique_numa_ids()) {
      name = std::string(NAME) ;
      benchmark::RegisterBenchmark(name.c_str(), NCCL_ops_reduce)->SMALL_ARGS()->UseManualTime();
     // name = std::string(NAME) + "_flush/" + std::to_string(numa_id) + "/" + std::to_string(cuda_id);
     // benchmark::RegisterBenchmark(name.c_str(), Comm_NUMAMemcpy_GPUToHost, numa_id, cuda_id, true)->SMALL_ARGS()->UseManualTime();
    //}
 // }
}

SCOPE_REGISTER_AFTER_INIT(registerer);

//BENCHMARK(NCCL_ops_reduce)->Apply(ArgsCountGpuGpuGpuGpu)->UseManualTime();

