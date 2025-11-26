//
// Created by zacha on 01-11-25.
//

#ifndef NEWIMPLEMENTATION_OPTIMIZER_FACTORY_H
#define NEWIMPLEMENTATION_OPTIMIZER_FACTORY_H
#include "optimizer.h"
#include "../training_component.h"
#include "../multi-threading.h"

enum optimizers {
    SIMPLE,
    FREE_MOMENTUM,
    PROPORTIONAL_MOMENTUM,
    SIMPLIFIED_NESTEROV,
    RMS_PROP,
    ADAM,
    ADAGRAD,
    ADADELTA,
    ADAMAX,
    ADAMW
};

extern tc_metadata opt_metadata[];

extern optimizer* cnstr_optimizer(int type, tc_cnstr_args args);
extern optimizer* cnstr_mt_optimizer(int type, tc_cnstr_args args, thread_pool* pool, int parallelCount);
extern char* get_opt_name(int type);

#endif //NEWIMPLEMENTATION_OPTIMIZER_FACTORY_H