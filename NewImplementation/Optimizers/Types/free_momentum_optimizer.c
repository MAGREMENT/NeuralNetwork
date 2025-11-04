//
// Created by zacha on 29-10-25.
//

#include "free_momentum_optimizer.h"

#include <stdlib.h>

static void apply_gradients(const optimizer* opt, double* to, const double* gradients, const int count, const optimizer_args args) {
    double* velocities = ((double**)args.state)[args.layerIndex];
    const double momentum = *(double*)opt->params;

    for (int i = 0; i < count; i++) {
        const double velocity = velocities[i] * momentum + gradients[i];

        velocities[i] = velocity;
        to[i] -= velocity * args.learning_rate;
    }
}

optimizer_vtable fm_vtable = {cnstr_gradient_buffers_state, free_gradient_buffers_state, apply_gradients, free_base_opt};

inline optimizer* cnstr_free_momentum_optimizer(const double momentum) {
    optimizer* opt = malloc(sizeof(optimizer));
    double* d = malloc(sizeof(double));
    *d = momentum;

    opt->params = d;
    opt->vtable = &fm_vtable;

    return opt;
}