//
// Created by zacha on 28-10-25.
//

#include "cosine_decay_scheduler.h"

#include <math.h>
#include <stdlib.h>

typedef struct cosine_decay_params {
    double ending;
    int span;
} cosine_decay_params;

static double schedule(const scheduler* sch, const double learningRate, const int iteration) {
    const cosine_decay_params* p = sch->params;
    return p->ending + 0.5 * (learningRate - p->ending) * (1 + cos(M_PI * iteration / p->span));
}

scheduler_vtable cds_vtable = {schedule, free_def_scheduler};

inline scheduler* cnstr_cosine_decay_scheduler(const double endingLearningRate, const int iterationSpan) {
    scheduler* sch = malloc(sizeof(scheduler));
    cosine_decay_params* d = malloc(sizeof(cosine_decay_params));
    d->ending = endingLearningRate;
    d->span = iterationSpan;

    sch->params = d;
    sch->vtable = &cds_vtable;

    return sch;
}