//
// Created by zacha on 29-10-25.
//

#include "rms_prop_optimizer.h"

#include <math.h>
#include <stdlib.h>

#define EPSILON 1e-8

static void apply_gradients(const optimizer* opt, double* to, const double* gradients, const int count, const optimizer_args args) {
    double* velocities = ((double**)args.state)[args.layerIndex];
    const double decay = *(double*)opt->params;

    for (int i = 0; i < count; i++) {
        const double velocity = velocities[i] * decay + gradients[i] * gradients[i] * (1 - decay);

        velocities[i] = velocity;
        to[i] -= args.learning_rate * gradients[i] / sqrt(velocity + EPSILON);
    }
}

optimizer_vtable rms_vtable = {cnstr_gradient_buffers_state, free_gradient_buffers_state, apply_gradients, free_base_opt};

optimizer* cnstr_rms_prop_optimizer(const double decay) {
    optimizer* opt = malloc(sizeof(optimizer));
    double* d = malloc(sizeof(double));
    *d = decay;

    opt->params = d;
    opt->vtable = &rms_vtable;

    return opt;
}