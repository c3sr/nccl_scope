# nccl|Scope

This is the nccl benchmark plugin for the [Scope](github.com/rai-project/scopes) benchmark project.

NCCL benchmarks for [C3SR Scope](https://github.com/c3sr/scope).

## Contributors

* [Carl Pearson](mailto:pearson@illinois.edu)
* [Sarah Hashash](mailto:hashash2@illinois.edu)

## Documentation

See the `docs` folder for a description of the benchmarks.

## Arguments

All support this arguemnt: 

* Number of GPUs
     + `-g, --ngpus` Default: 1. 



## Changelog

### v0.1.1

  * Add changelog

### v0.1.0

  * Bandwidth benchmarks for `allGather`, `allReduce`, `broadcast`, `reduce`, `reduceScatter`.
  * Disable NCCL|Scope by default (enable with `-DENABLE_NCCL=1`).



