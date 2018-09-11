#include "scope/init/init.hpp"
#include "scope/init/flags.hpp"
#include "scope/utils/commandlineflags.hpp"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>

#include "flags.hpp"

//DECLARE_int32(ngpu);
//DEFINE_int32(ngpu, 1, "number of gpus");
//DECLARE_string(ops);
//DEFINE_string(ops, 1, "reduction operations");

int nccl_scope_init(int argc, char *const *argv) {
for (int i = 1; i < argc; ++i) {
    utils::ParseInt32Flag(argv[i], "ngpu", &FLAG(ngpu)); // --ngpu=70 causes *ngpu = 70
//    utils::ParseStringFlag(argv[i], "ops//", &FLAG(ops));
  }
//for (const auto &e : FLAG(ngpu)) {
//    LOG(debug, "User requested NUMA node " + std::to_string(e));
//  }
//    (void) argc;
//    (void) argv;
return 0;
}

SCOPE_INIT(nccl_scope_init);
