//
// Created by zacha on 28-10-25.
//

#include "inverse_decay_scheduler.h"

#include <stdlib.h>

static double schedule(const scheduler* sch, const double learningRate, const int iteration) {
    const double decay = *(double*)sch->params;

    return learningRate / (1 - decay * iteration);
}

scheduler_vtable ivds_vtable = {schedule, free_def_scheduler};

inline scheduler* cnstr_inverse_decay_scheduler(const double decay) {
    scheduler* sch = malloc(sizeof(scheduler));
    double* d = malloc(sizeof(double));
    *d = decay;

    sch->params = d;
    sch->vtable = &ivds_vtable;

    return sch;
}