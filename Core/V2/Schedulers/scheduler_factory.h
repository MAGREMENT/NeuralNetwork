//
// Created by zacha on 01-11-25.
//

#ifndef NEWIMPLEMENTATION_SCHEDULER_FACTORY_H
#define NEWIMPLEMENTATION_SCHEDULER_FACTORY_H
#include "scheduler.h"
#include "../training_component.h"

#define SCHEDULER_COUNT 5

enum schedulers {
    CONSTANT,
    ITERATION_DECAY,
    EXPONENTIAL_DECAY,
    INVERSE_DECAY,
    COSINE_DECAY
};

extern tc_metadata sch_metadata[];

extern scheduler* cnstr_scheduler(int type, tc_cnstr_args args);

#endif //NEWIMPLEMENTATION_SCHEDULER_FACTORY_H