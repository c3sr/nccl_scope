#include "scope/init/init.hpp"
#include "scope/init/flags.hpp"
#include "scope/utils/commandlineflags.hpp"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>

#include "flags.hpp"


void nccl_before_init() {
  RegisterOpt(  clara::Opt(FLAG(ngpu), "ngpu")["-g"]["--ngpu"]("add choose gpus (Nccl|Scope)")  );
}


int nccl_scope_init() {
  return 0;
}
SCOPE_REGISTER_BEFORE_INIT(nccl_before_init);
SCOPE_REGISTER_INIT(nccl_scope_init);
