//
// Created by zacha on 01-11-25.
//

#include "optimizer_factory.h"

#include <stddef.h>

#include "Types/adadelta_optimizer.h"
#include "Types/adagrad_optimizer.h"
#include "Types/adamax_optimizer.h"
#include "Types/adamw_optimizer.h"
#include "Types/adam_optimizer.h"
#include "Types/simple_optimizer.h"
#include "Types/free_momentum_optimizer.h"
#include "Types/simplified_nesterov_optimizer.h"
#include "Types/proportional_momentum_optimizer.h"
#include "Types/rms_prop_optimizer.h"

tc_metadata opt_metadata[] = {
    {"Simple", TCT_NONE},
    {"Free Momentum", TCT_DOUBLE},
    {"Proportional Momentum", TCT_DOUBLE},
    {"Simplified Nesterov", TCT_DOUBLE},
    {"RMSProp", TCT_DOUBLE},
    {"Adam", TCT_DOUBLE2},
    {"AdaGrad", TCT_NONE},
    {"AdaDelta", TCT_DOUBLE},
    {"AdaMax", TCT_DOUBLE2},
    {"AdamW", TCT_DOUBLE3},
};

optimizer* cnstr_optimizer(const int type, const tc_cnstr_args args) {
    switch (type) {
        case SIMPLE : return cnstr_simple_optimizer();
        case FREE_MOMENTUM: return cnstr_free_momentum_optimizer(args.d);
        case PROPORTIONAL_MOMENTUM : return cnstr_proportional_momentum_optimizer(args.d);
        case SIMPLIFIED_NESTEROV : return cnstr_simplified_nesterov_optimizer(args.d);
        case RMS_PROP : return cnstr_rms_prop_optimizer(args.d);
        case ADAM : return cnstr_adam_optimizer(args.d2.d1, args.d2.d2);
        case ADAGRAD : return cnstr_adagrad_optimizer();
        case ADADELTA : return cnstr_adadelta_optimizer(args.d);
        case ADAMAX : return cnstr_adamax_optimizer(args.d2.d1, args.d2.d2);
        case ADAMW : return cnstr_adamw_optimizer(args.d3.d1, args.d3.d2, args.d3.d3);
        default : return NULL;
    }
}

optimizer* cnstr_mt_optimizer(const int type, const tc_cnstr_args args, thread_pool* pool, const int parallelCount) {
    switch (type) {
        case ADAM : return cnstr_mt_adam_optimizer(args.d2.d1, args.d2.d2, pool, parallelCount);
        default : return NULL;
    }
}
