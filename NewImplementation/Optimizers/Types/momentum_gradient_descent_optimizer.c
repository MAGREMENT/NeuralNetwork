//
// Created by zacha on 29-10-25.
//

#include "momentum_gradient_descent_optimizer.h"

#include <stdlib.h>

static void apply_gradients(const optimizer* opt, double* to, const double* gradients, const int count, const optimizer_args args) {
    double* velocities = ((double**)args.state)[args.layerIndex];
    const double momentum = *(double*)opt->params;

    for (int i = 0; i < count; i++) {
        const double velocity = velocities[i] * momentum + gradients[i]; //TODO with averaging

        velocities[i] = velocity;
        to[i] -= velocity * args.learning_rate;
    }
}

static void free_mgdo(optimizer* opt) {
    free(opt->params);
    free(opt);
}

optimizer_vtable mgdo_vtable = {cnstr_gradient_buffers_state, free_gradient_buffers_state, apply_gradients, free_mgdo};

inline optimizer* cnstr_momentum_gradient_descent_optimizer(const double momentum) {
    optimizer* opt = malloc(sizeof(optimizer));
    double* d = malloc(sizeof(double));
    *d = momentum;

    opt->params = d;
    opt->vtable = &mgdo_vtable;

    return opt;
}