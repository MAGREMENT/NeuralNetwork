//
// Created by zacha on 04-11-25.
//

#include "adagrad_optimizer.h"

#include <math.h>
#include <stdlib.h>

#define EPSILON 1e-8

static void apply_gradients(const optimizer* opt, double* to, const double* gradients, const int count, const optimizer_args args) {
    double* accumulated = ((double**)args.state)[args.layerIndex];

    for (int i = 0; i < count; i++) {
        accumulated[i] += gradients[i] * gradients[i];
        to[i] -= gradients[i] * args.learning_rate / sqrt(accumulated[i] + EPSILON);
    }
}

optimizer_vtable adagrad_vtable = {cnstr_gradient_buffers_state, free_gradient_buffers_state, apply_gradients, free_empty_opt};

inline optimizer* cnstr_adagrad_optimizer() {
    optimizer* opt = malloc(sizeof(optimizer));

    opt->params = NULL;
    opt->vtable = &adagrad_vtable;

    return opt;
}