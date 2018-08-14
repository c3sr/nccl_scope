# Unified Memory NCCL Bandwidth

Comm|Scope defines 5 microbenchmarks to measure unified memory coherence bandwidth.
These benchmarks may be listed with the argument

    --benchmark_filter="NCCL_ops"

## Implementations

|Benchmarks|Description|Argument Format|
|-|-|-|
| `NCCL_ops_allReduce` | allReduce | `log2 size / src GPUs / dst GPUs` |
| `NCCL_ops_reduce` | reduce | `log2 size / src GPUs / dst GPU` |
| `NCCL_ops_allGather` | allGather | `log2 size / src GPUs / dst GPUs` |
| `NCCL_ops_reduceScatter` | reduceScatter | `log2 size / src GPUs / dst GPUs` |
| `NCCL_ops_broadcast` | broadcast | `log2 size / src GPU / dst GPUs` |

## allReduce, broadcast, and reduce GPU Technique

To perform reductions/copies on data across devices, the benchmark setup phase looks like this

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
## reduceScatter GPU Technique

To perform reductions/copies on data across devices, the benchmark setup phase looks like this

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
    cudaMalloc(sendbuff + i, size * sizeof(float) * nDev)
    cudaMalloc(recvbuff + i, size * sizeof(float))
    cudaMemset(sendbuff[i], 1, size * sizeof(float) * nDev)
    cudaMemset(recvbuff[i], 0, size * sizeof(float))
    cudaStreamCreate(s+i)
    cudaEventCreate(starts+i)
    cudaEventCreate(stops+i)
    }
end loop
```
## allGather GPU Technique

To perform reductions/copies on data across devices, the benchmark setup phase looks like this

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
    cudaMalloc(recvbuff + i, size * sizeof(float)*nDev)
    cudaMemset(sendbuff[i], 1, size * sizeof(float))
    cudaMemset(recvbuff[i], 0, size * sizeof(float)*nDev)
    cudaStreamCreate(s+i)
    cudaEventCreate(starts+i)
    cudaEventCreate(stops+i)
    }
end loop
```

## The benchmark loop 

```
loop (state)
  cudaEventRecord(start, NULL)  

    ncclGroupStart()
    loop(nDev)
        cudaEventRecord(starts[i],s[i])
	ncclOperation()
        cudaEventRecord(stops[i],s[i])
    end loop
    ncclGroupEnd()

  cudaEventRecord(stop, NULL)
    loop(nDev)
        cudaStreamSynchronize(s[i])
    end loop  
    cudaEventSynchronize(stop)

  //timing
  state.PauseTiming()
  float msecTotal = 0.0f
  cudaEventElapsedTime(&msecTotal, start, stop)
  state.SetIterationTime(msecTotal/ 1000)
  state.ResumeTiming()

end loop
```

