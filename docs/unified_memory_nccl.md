# Unified Memory NCCL Bandwidth

Comm|Scope defines 5 microbenchmarks to measure unified memory coherence bandwidth.
These benchmarks may be listed with the argument

    --benchmark_filter="NCCL_function"

## Implementations

|Benchmarks|Description|Argument Format|
|-|-|-|
| `NCCL_function_ALLREDUCE` | allReduce | `log2 size / src GPUs / dst GPUs` |
| `NCCL_function_REDUCE` | reduce | `log2 size / src GPUs / dst GPU` |
| `NCCL_function_ALLGATHER` | allGather | `log2 size / src GPUs / dst GPUs` |
| `NCCL_function_REDUCESCATTER` | reduceScatter | `log2 size / src GPUs / dst GPUs` |
| `NCCL_function_BROADCAST` | broadcast | `log2 size / src GPU / dst GPUs` |

## allReduce GPU Technique

To perform reductions on data across devices, the benchmark setup phase looks like this

```
// communicator setup 
  ncclComm_t comms

//allocate and initalize device buffers
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
  cudaEvent_t* starts = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*nDev);
  cudaEvent_t* stops = (cudaEvent_t*)malloc(sizeof(cudaEvent_t)*nDev);

for (int i = 0; i < nDev; ++i) {
    cudaSetDevice(i)
    cudaMalloc(sendbuff + i, size * sizeof(float))
    cudaMalloc(recvbuff + i, size * sizeof(float))
    cudaMemset(sendbuff[i], 1, size * sizeof(float))
    cudaMemset(recvbuff[i], 0, size * sizeof(float))
    cudaStreamCreate(s+i)
    cudaEventCreate(starts+i)
    cudaEventCreate(stops+i)
    }
end loop
```

For a device -> host transfer, the setup and benchmark loop looks like this

```
// device-to-host setup
cudaSetDevice(src)
numa_bind(dst)
cudaMallocManaged(&ptr)

// device-to-host benchmark loop
loop (state)
    // move pages to src
    cudaMemPrefetchAsync(ptr, src)
    cudaDeviceSynchronize(src)
    // execute workload on cpu
    state.resumeTiming()
    write_cpu(ptr)
    state.stopTiming()
end loop
```

