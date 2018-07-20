# nccl|Scope

This is the nccl benchmark plugin for the [Scope](github.com/rai-project/scopes) benchmark project.

## Adding the plugin to Scope

The plugin is expected to be loaded as an optional git submodule in the Scope repo, to be included in the build via `add_subdirectory(<scope path>)` in Scope's `CMakeLists.txt`.
This means that the plugin will inherit any variables Scope defines.

The plugin should export any libraries it requires so that Scope can link against them during the build step.

## Scope Utilities

The plugin may/should make use of utilities provided by Scope in scope

## Scope Initialization

Scope allows plugins to register initialization callbacks in scope/src/init.hpp.

Callbacks are `void (*fn)(int argc, char **argv)` functions that will be passed the command line flags that Scope is executed with.
Callbacks can be registered with the INIT() macro:

```cpp
// plugin/src/init.cpp
#include "scope/init/init.hpp

void plugin_init(int argc, char **argv) {
    (void) argc;
    (void) argv;
}

INIT(pugin_init);
```

Scope does not guarantee any ordering for callback execution.

## Structure



### `CMakeLists.txt`

The plugin `CMakeLists.txt` should
* `sugar_include` the plugin sources
* `add_library` to create a plugin library
* find any required packages needed by the plugin
* link any required libraries with the PUBLIC keyword so that Scope is also linked with them
```
target_link_libraries(example_scope PUBLIC required-library)
```

Scope provides a python script for generating `sugar.cmake` files.
It should be invoked like this whenever source files are added or moved in the plugin:

    $ [scope-dir]/tools/generate_sugar_files.py --top [plugin-dir]/src --var plugin-name

This will cause `plugin-name_SOURCES` and `plugin-name_CUDA_SOURCES` to be defined.
These are most likely the variables that should be expanded when using `add_library` in the plugin `CMakeLists.txt`.

### `docs`

The plugin `docs` folder should describe all of the benchmarks created by the plugin.


