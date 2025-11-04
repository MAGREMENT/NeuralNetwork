//
// Created by zacha on 04-11-25.
//

#include "adadelta_optimizer.h"

#include <stdlib.h>
#include <math.h>

#define EPSILON 1e-8 //TODO for all epsilons, add it as parameter

static void apply_gradients(const optimizer* opt, double* to, const double* gradients, const int count, const optimizer_args args) {
    double* avg_grads = ((double***)args.state)[0][args.layerIndex];
    double* avg_params = ((double***)args.state)[1][args.layerIndex];
    const double decay = *(double*)opt->params;

    for (int i = 0; i < count; i++) {
        avg_grads[i] = decay * avg_grads[i] + (1 - decay) * gradients[i] * gradients[i];

        const double param = -sqrt(avg_params[i] + EPSILON) * gradients[i] / sqrt(avg_grads[i] + EPSILON);

        avg_params[i] = decay * avg_params[i] + (1 - decay) * param * param;

        to[i] += param;
    }
}

optimizer_vtable adadelta_vtable = {cnstr_double_gradient_buffers_state, free_double_gradient_buffers_state, apply_gradients, free_base_opt};

inline optimizer* cnstr_adadelta_optimizer(const double decay) {
    optimizer* opt = malloc(sizeof(optimizer));
    double* d = malloc(sizeof(double));
    *d = decay;

    opt->params = d;
    opt->vtable = &adadelta_vtable;

    return opt;
}