//
// Created by zacha on 01-11-25.
//

#ifndef NEWIMPLEMENTATION_SCHEDULER_FACTORY_H
#define NEWIMPLEMENTATION_SCHEDULER_FACTORY_H
#include "scheduler.h"

enum schedulers {
    CONSTANT,
    ITERATION_DECAY,
    EXPONENTIAL_DECAY,
    INVERSE_DECAY,
    COSINE_DECAY
};

typedef struct double_int {
    double d;
    int i;
} double_int;

typedef union scheduler_cnstr_args {
    double value;
    double_int di_value;
} scheduler_cnstr_args;

scheduler* cnstr_scheduler(int type, scheduler_cnstr_args args);

#endif //NEWIMPLEMENTATION_SCHEDULER_FACTORY_H