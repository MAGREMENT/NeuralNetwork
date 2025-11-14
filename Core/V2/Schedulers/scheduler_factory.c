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

scheduler* cnstr_scheduler(const int type, const scheduler_cnstr_args args) {
    switch (type) {
        case CONSTANT : return cnstr_constant_scheduler();
        case ITERATION_DECAY : return cnstr_iteration_decay_scheduler(args.di_value.d, args.di_value.i);
        case EXPONENTIAL_DECAY : return cnstr_exponential_decay_scheduler(args.value);
        case INVERSE_DECAY : return cnstr_inverse_decay_scheduler(args.value);
        case COSINE_DECAY : return cnstr_cosine_decay_scheduler(args.di_value.d, args.di_value.i);
        default : return NULL;
    }
}
