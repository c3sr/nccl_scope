#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <nccl.h> //don't have this yet

#include <cuda_runtime.h>

#include "scope/init/init.hpp"

#include "SingleDevices/args.hpp"

#define NAME "SINGLE/nccl/GPUGPU"
//combines data from all sender and delvers to all participants 
static void SINGLE_nccl_ALLREDUCE(benchmark::State &state) {

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

  // There are two copies, one gpu0 -> gpu1, one gpu1 -> gpu0

  // Create One stream per copy
  cudaStream_t stream1, stream2;
  std::vector<cudaStream_t> streams = {stream1, stream2};
  cudaStreamCreate(&streams[0]);
  cudaStreamCreate(&streams[1]);


  // Start and stop events for each copy
  cudaEvent_t start1, start2, stop1, stop2;
  std::vector<cudaEvent_t> starts = {start1, start2};
  std::vector<cudaEvent_t> stops = {stop1, stop2};
  cudaEventCreate(&starts[0]);
  cudaEventCreate(&starts[1]);
  cudaEventCreate(&stops[0]);
  cudaEventCreate(&stops[1]);

  // Source and destination for each copy
  std::vector<char *> srcs;
  std::vector<char *> dsts;

  // create a source and destination allocation for first copy
  char *ptr;
  if (PRINT_IF_ERROR(cudaSetDevice(gpu0))) {
    state.SkipWithError(NAME " failed to set device");
    return;
  }
  if (PRINT_IF_ERROR(cudaMalloc(&ptr, bytes))) {
    state.SkipWithError(NAME " failed to perform cudaMalloc");
    return;
  }
  srcs.push_back(ptr);
  if (PRINT_IF_ERROR(cudaMemset(ptr, 0, bytes))) {
    state.SkipWithError(NAME " failed to perform src cudaMemset");
    return;
  }
  cudaError_t err = cudaDeviceEnablePeerAccess(gpu1, 0);
  if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
    state.SkipWithError(NAME " failed to ensure peer access");
    return;
  }
  if (PRINT_IF_ERROR(cudaSetDevice(gpu1))) {
    state.SkipWithError(NAME " failed to set device");
    return;
  }
  if (PRINT_IF_ERROR(cudaMalloc(&ptr, bytes))){
    state.SkipWithError(NAME " failed to perform cudaMalloc");
    return;
  }
  dsts.push_back(ptr);
  if(PRINT_IF_ERROR(cudaMemset(ptr, 0, bytes))){
    state.SkipWithError(NAME " failed to perform dst cudaMemset");
    return;
  }
  err = cudaDeviceEnablePeerAccess(gpu0, 0);
  if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
    state.SkipWithError(NAME " failed to ensure peer access");
    return;
  }
  // create a source and destination for second copy
  if (PRINT_IF_ERROR(cudaSetDevice(gpu1))) {
    state.SkipWithError(NAME " failed to set device");
    return;
  }
  if (PRINT_IF_ERROR(cudaMalloc(&ptr, bytes))) {
    state.SkipWithError(NAME " failed to perform cudaMalloc");
    return;
  }
  srcs.push_back(ptr);
  if (PRINT_IF_ERROR(cudaMemset(ptr, 0, bytes))) {
    state.SkipWithError(NAME " failed to perform src cudaMemset");
    return;
  }
  err = cudaDeviceEnablePeerAccess(gpu0, 0);
  if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
    state.SkipWithError(NAME " failed to ensure peer access");
    return;
  }
  if (PRINT_IF_ERROR(cudaSetDevice(gpu0))) {
    state.SkipWithError(NAME " failed to set device");
    return;
  }
  if (PRINT_IF_ERROR(cudaMalloc(&ptr, bytes))){
    state.SkipWithError(NAME " failed to perform cudaMalloc");
    return;
  }
  dsts.push_back(ptr);
  if(PRINT_IF_ERROR(cudaMemset(ptr, 0, bytes))){
    state.SkipWithError(NAME " failed to perform dst cudaMemset");
    return;
  }
  err = cudaDeviceEnablePeerAccess(gpu1, 0);
  if (cudaSuccess != err && cudaErrorPeerAccessAlreadyEnabled != err) {
    state.SkipWithError(NAME " failed to ensure peer access");
    return;
  }


  assert(starts.size() == stops.size());
  assert(streams.size() == starts.size());
  assert(srcs.size() == dsts.size());
  assert(streams.size() == srcs.size());

  for (auto _ : state) {

    // Start all copies
    for (size_t i = 0; i < streams.size(); ++i) {
      auto start = starts[i];
      auto stop = stops[i];
      auto stream = streams[i];
      auto src = srcs[i];
      auto dst = dsts[i];
      if(PRINT_IF_ERROR(cudaEventRecord(start, stream))) {
        state.SkipWithError(NAME " failed to record start event");
        return;
      }
      //where all reduce happends
      if(PRINT_IF_ERROR(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream))) {
        state.SkipWithError(NAME " failed to start cudaMemcpyAsync");
        return;
      }
      if(PRINT_IF_ERROR(cudaEventRecord(stop, stream))) {
        state.SkipWithError(NAME " failed to record stop event");
        return;
      }
    }

    // Wait for all copies to finish
    for (auto s : stops) {
      if (PRINT_IF_ERROR(cudaEventSynchronize(s))) {
        state.SkipWithError(NAME " failed to synchronize");
        return;
      }
    }

    // Find the longest time between any start and stop
    float maxMillis = 0;
    for (const auto start : starts) {
      for (const auto stop : stops) {
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

  for (auto src : srcs) {
    cudaFree(src);
  }
  for (auto dst : dsts) {
    cudaFree(dst);
  }
}

BENCHMARK(SINGLE_Nccl_ALLReduce)->Apply(ArgsCountGpuGpuNoSelf)->UseManualTime();


#include <nccl.h>
ncclComm_t comm[4];
ncclCommInitAll(comm, 4, {0, 1, 2, 3});
foreach g in (GPUs) { // or foreach thread
cudaSetDevice(g);
double *d_send,
*d_recv;
// allocate d_send, d_recv; fill d_send with data
ncclAllReduce(d_send, d_recv, N, ncclDouble, ncclSum, comm[g], stream[g]);
// consume d_recv
}
