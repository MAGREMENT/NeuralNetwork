//
// Created by zacha on 01-11-25.
//

#include "scheduler_factory.h"

#include <stddef.h>

#include "Types/constant_scheduler.h"
#include "Types/cosine_decay_scheduler.h"
#include "Types/exponential_decay_scheduler.h"
#include "Types/inverse_decay_scheduler.h"
#include "Types/iteration_decay_scheduler.h"

tc_metadata sch_metadata[] = {
    {"Constant", TCT_NONE},
    {"Iteration Decay", TCT_DOUBLE_INT},
    {"Exponential Decay", TCT_DOUBLE},
    {"Inverse Decay", TCT_DOUBLE},
    {"Cosine Decay", TCT_DOUBLE_INT}
};

scheduler* cnstr_scheduler(const int type, const tc_cnstr_args args) {
    switch (type) {
        case CONSTANT : return cnstr_constant_scheduler();
        case ITERATION_DECAY : return cnstr_iteration_decay_scheduler(args.di.d, args.di.i);
        case EXPONENTIAL_DECAY : return cnstr_exponential_decay_scheduler(args.d);
        case INVERSE_DECAY : return cnstr_inverse_decay_scheduler(args.d);
        case COSINE_DECAY : return cnstr_cosine_decay_scheduler(args.di.d, args.di.i);
        default : return NULL;
    }
}
