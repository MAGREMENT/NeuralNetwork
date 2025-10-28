//
// Created by zacha on 28-10-25.
//

#include "iteration_decay_scheduler.h"

#include <math.h>
#include <stdlib.h>

typedef struct ids_params {
    double decay;
    int stepSize;
} ids_params;

static double schedule(const scheduler* sch, const double learningRate, const int iteration) {
    const double decay = ((ids_params*)sch->params)->decay;
    const double stepSize = ((ids_params*)sch->params)->stepSize;

    return learningRate * pow(1 - decay, iteration / stepSize);
}

scheduler_vtable ids_vtable = {schedule, free_def_scheduler};

inline scheduler* cnstr_iteration_decay_scheduler(const double decay, const int stepSize) {
    scheduler* sch = malloc(sizeof(scheduler) * 2);
    ids_params* d = malloc(sizeof(ids_params));
    d->decay = decay;
    d->stepSize = stepSize;

    sch->params = d;
    sch->vtable = &ids_vtable;

    return sch;
}