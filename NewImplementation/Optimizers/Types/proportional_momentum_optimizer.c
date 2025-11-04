//
// Created by zacha on 03-11-25.
//

#include "proportional_momentum_optimizer.h"

#include <stdlib.h>

static void apply_gradients(const optimizer* opt, double* to, const double* gradients, const int count, const optimizer_args args) {
    double* velocities = ((double**)args.state)[args.layerIndex];
    const double momentum = *(double*)opt->params;

    for (int i = 0; i < count; i++) {
        const double velocity = velocities[i] * momentum + gradients[i] * (1 - momentum);

        velocities[i] = velocity;
        to[i] -= velocity * args.learning_rate;
    }
}

optimizer_vtable pm_vtable = {cnstr_gradient_buffers_state, free_gradient_buffers_state, apply_gradients, free_base_opt};

inline optimizer* cnstr_proportional_momentum_optimizer(const double momentum) {
    optimizer* opt = malloc(sizeof(optimizer));
    double* d = malloc(sizeof(double));
    *d = momentum;

    opt->params = d;
    opt->vtable = &pm_vtable;

    return opt;
}