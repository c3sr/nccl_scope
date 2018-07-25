#include "scope/init/init.hpp

void plugin_init(int argc, char **argv) {
    (void) argc;
    (void) argv;
}

INIT(nccl_scope_init);
