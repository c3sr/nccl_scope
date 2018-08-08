# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED NCCL_SCOPE_SRC_SINGLEPROCESS_SUGAR_CMAKE_)
  return()
else()
  set(NCCL_SCOPE_SRC_SINGLEPROCESS_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)

sugar_files(
    nccl_HEADERS
    args.hpp
)

sugar_files(
    nccl_SOURCES
    allGather.cpp
    charallReduce.cpp
    exallReduce.cpp
    reduceScatter.cpp
)

