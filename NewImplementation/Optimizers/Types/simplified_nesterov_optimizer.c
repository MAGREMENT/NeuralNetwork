//
// Created by zacha on 29-10-25.
//

#include "simplified_nesterov_optimizer.h"

#include <stdlib.h>

static void apply_gradients(const optimizer* opt, double* to, const double* gradients, const int count, const optimizer_args args) {
    double* velocities = ((double**)args.state)[args.layerIndex];
    const double momentum = *(double*)opt->params;

    for (int i = 0; i < count; i++) {
        const double velocity = velocities[i] * momentum - args.learning_rate * gradients[i];

        to[i] = to[i] - velocities[i] * momentum + (1 + momentum) * velocity;
        velocities[i] = velocity;
    }
}

optimizer_vtable nest_vtable = {cnstr_gradient_buffers_state, free_gradient_buffers_state, apply_gradients, free_base_opt};

inline optimizer* cnstr_simplified_nesterov_optimizer(const double momentum) {
    optimizer* opt = malloc(sizeof(optimizer));
    double* d = malloc(sizeof(double));
    *d = momentum;

    opt->params = d;
    opt->vtable = &nest_vtable;

    return opt;
}