cmake_minimum_required(VERSION 3.8)

# Look for sugar.cmake files in the plugin src directory
sugar_include("src")

# The plugin is a static library that Scope will be linked with
add_library(nccl_scope ${nccl_SOURCES} ${nccl_CUDA_SOURCES} )
target_include_directories(nccl_scope PRIVATE 
    ${SRC_DIR}
    ${THIRDPARTY_DIR}
    ${TOP_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    "src"
)

target_compile_features(nccl_scope PUBLIC cxx_std_11)
set_property(TARGET nccl_scope PROPERTY CUDA_STANDARD 11)
target_link_libraries(nccl_scope PRIVATE benchmark::benchmark)

#Nccl
#find_package(NCCL)
#if(NCCL_FOUND)
#  target_compile_definitions(nccl_scope PRIVATE -DUSE_NCCL=1) 
#  target_link_libraries(nccl_scope PUBLIC NCCL)
#else()
#  target_compile_definitions(nccl_scope PRIVATE -DUSE_NCCL=0)
#  message(WARNING "didn't find NCCL, benchmarks will not be availabl.")
#endif(NCCL_FOUND)
target_compile_definitions(nccl_scope PRIVATE -DCUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR})
target_compile_definitions(nccl_scope PRIVATE -DCUDA_VERSION_MINOR=${CUDA_VERSION_MINOR})

MESSAGE(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
string(REPLACE ";" " " CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
MESSAGE(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
