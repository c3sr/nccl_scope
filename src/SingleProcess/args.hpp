#include <cstdint>
#include <cstdlib>

inline static size_t popcount(uint64_t u) {
  return __builtin_popcount(u);
}

inline static bool is_set(uint64_t bits, size_t i) {
  return (uint64_t(1) << i) & bits;
}

inline static void ArgsCountGpuGpuGpuGpu(benchmark::internal::Benchmark* b) {

  int n;
  cudaError_t err = cudaGetDeviceCount(&n);
  if (PRINT_IF_ERROR(cudaGetDeviceCount(&n))) {
    exit(1);
  }
/*
  for (int gpu0 = 0; gpu0 < n; ++gpu0) {
    for (int gpu1 = 0; gpu1 < n; ++gpu1) {
      for(int gpu2 = 0; gpu2 < n; ++gpu2){
        for(int gpu3 = 0; gpu3 < n; ++gpu3){
               for (int j = 8; j < 31; ++j) {
                 b->Args({j, gpu0, gpu1, gpu2, gpu3});
               }
        }
      }
    }
  }
*/

  for (int gpu0 = 0; gpu0 < n; ++gpu0){
   for(int gpu1 = 0; gpu1 < n; ++gpu1){
     for(int j = 8; j <31; ++j){
      int gpu2 = 0;
      int gpu3 = 0;
      b-> Args({j,gpu0,gpu1, gpu2, gpu3});
      }
   }
  }


}

