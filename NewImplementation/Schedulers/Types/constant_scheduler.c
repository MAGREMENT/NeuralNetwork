//
// Created by zacha on 28-10-25.
//

#include "constant_scheduler.h"

#include <stdlib.h>

static double schedule(const scheduler* sch, const double learningRate, const int iteration) {
    return learningRate;
}

scheduler_vtable cs_vtable = {schedule, free_empty_scheduler};

scheduler* cnstr_constant_scheduler() {
    scheduler* sch = malloc(sizeof(scheduler));

    sch->params = NULL;
    sch->vtable = &cs_vtable;

    return sch;
}