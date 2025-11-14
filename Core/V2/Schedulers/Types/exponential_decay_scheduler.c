//
// Created by zacha on 28-10-25.
//

#include "exponential_decay_scheduler.h"

#include <math.h>
#include <stdlib.h>

static double schedule(const scheduler* sch, const double learningRate, const int iteration) {
    const double decay = *(double*)sch->params;

    return learningRate * exp(-decay * iteration);
}

scheduler_vtable eds_vtable = {schedule, free_def_scheduler};

scheduler* cnstr_exponential_decay_scheduler(const double decay) {
    scheduler* sch = malloc(sizeof(scheduler));
    double* d = malloc(sizeof(double));
    *d = decay;

    sch->params = d;
    sch->vtable = &eds_vtable;

    return sch;
}