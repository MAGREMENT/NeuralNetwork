//
// Created by zacha on 29-10-25.
//

#include "nesterov_optimizer.h"

#include <stdlib.h>

#define PREDICTION_FACTOR 1

static void apply_gradients(const optimizer* opt, double* to, const double* gradients, const int count, const optimizer_args args) {
    double* velocities = ((double**)args.state)[args.layerIndex];
    const double momentum = *(double*)opt->params;

    for (int i = 0; i < count; i++) {
        const double velocity = velocities[i] * momentum + gradients[i];

        velocities[i] = velocity;
        to[i] -= (velocity * momentum + gradients[i] * PREDICTION_FACTOR) * args.learning_rate;
    }
}

static void free_nest(optimizer* opt) {
    free(opt->params);
    free(opt);
}

optimizer_vtable nest_vtable = {cnstr_gradient_buffers_state, free_gradient_buffers_state, apply_gradients, free_nest};

inline optimizer* cnstr_nesterov_optimizer(const double momentum) {
    optimizer* opt = malloc(sizeof(optimizer));
    double* d = malloc(sizeof(double));
    *d = momentum;

    opt->params = d;
    opt->vtable = &nest_vtable;

    return opt;
}