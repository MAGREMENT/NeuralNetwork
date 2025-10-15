//
// Created by zacha on 01-10-25.
//

#include "learning_rate_sechduler.h"

#include <math.h>
#include <stdlib.h>

static void free_default_scheduler(learning_rate_scheduler* sch) {
    free(sch->params);
    free(sch);
}

static double constant_schedule(learning_rate_scheduler* sch, const double learningRate, int iteration) {
    return learningRate;
}

inline learning_rate_scheduler* constr_constant_scheduler() {
    learning_rate_scheduler* sch = malloc(sizeof(learning_rate_scheduler));

    sch->free = free;
    sch->schedule = constant_schedule;
    sch->alloc_to_hyper = NULL;

    return sch;
}

static learning_rate_scheduler* constr_single_double_param_scheduler(const double decay) {
    learning_rate_scheduler* sch = malloc(sizeof(learning_rate_scheduler));
    double* d = malloc(sizeof(double));
    *d = decay;

    sch->params = d;
    sch->free = free_default_scheduler;

    return sch;
}

//TODO introduce step size
static double iteration_schedule(learning_rate_scheduler* sch, const double learningRate, int iteration) {
    const double decay = *(double*)sch->params;
    return learningRate * pow(decay, iteration);
}

inline learning_rate_scheduler* constr_iteration_decay_scheduler(const double decay) {
    learning_rate_scheduler* sch = constr_single_double_param_scheduler(decay);
    sch->schedule = iteration_schedule;
    sch->alloc_to_hyper = NULL;

    return sch;
}

static double exponential_schedule(learning_rate_scheduler* sch, const double learningRate, int iteration) {
    const double decay = *(double*)sch->params;
    return learningRate * exp(-1 * decay * iteration);
}

inline learning_rate_scheduler* constr_exponential_decay_scheduler(const double decay) {
    learning_rate_scheduler* sch = constr_single_double_param_scheduler(decay);
    sch->schedule = exponential_schedule;
    sch->alloc_to_hyper = NULL;

    return sch;
}

static double inverse_schedule(learning_rate_scheduler* sch, const double learningRate, int iteration) {
    const double decay = *(double*)sch->params;
    return learningRate / (1 + decay * iteration);
}

inline learning_rate_scheduler* constr_inverse_decay_scheduler(const double decay) {
    learning_rate_scheduler* sch = constr_single_double_param_scheduler(decay);
    sch->schedule = inverse_schedule;
    sch->alloc_to_hyper = NULL;

    return sch;
}

typedef struct cosine_decay_params {
    double ending;
    int span;
} cosine_decay_params;

static double cosine_schedule(learning_rate_scheduler* sch, const double learningRate, int iteration) {
    cosine_decay_params* p = sch->params;
    return p->ending + 0.5 * (learningRate - p->ending) * (1 + cos(M_PI * iteration / p->span));
}

learning_rate_scheduler* constr_cosine_decay_scheduler(const double endingLr, const int iterationSpan) {
    learning_rate_scheduler* sch = malloc(sizeof(learning_rate_scheduler));
    cosine_decay_params* params = malloc(sizeof(cosine_decay_params));
    params->ending = endingLr;
    params->span = iterationSpan;

    sch->params = params;
    sch->free = free_default_scheduler;
    sch->schedule = cosine_schedule;
    sch->alloc_to_hyper = NULL;

    return sch;
}

