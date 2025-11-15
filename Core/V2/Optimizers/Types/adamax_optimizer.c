//
// Created by zacha on 15-11-25.
//

#include "adamax_optimizer.h"

#include <math.h>
#include <stdlib.h>

#define EPSILON 1e-8

static void apply_gradients(const optimizer* opt, double* to, const double* gradients, const int count, const optimizer_args args) {
    double* v1 = ((double***)args.state)[0][args.layerIndex];
    double* u2 = ((double***)args.state)[1][args.layerIndex];
    const double beta1 = ((double*)opt->params)[0];
    const double beta2 = ((double*)opt->params)[1];
    const double beta1t = 1 - pow(beta1, args.iteration);

    for (int i = 0; i < count; i++) {
        const double grad = gradients[i];
        v1[i] = beta1 * v1[i] + (1 - beta1) * grad;
        u2[i] = max(beta2 * u2[i], fabs(grad));

        const double m = v1[i] / beta1t;

        to[i] -= args.learning_rate * m / (u2[i] + EPSILON);
    }
}

optimizer_vtable adamax_vtable = {cnstr_double_gradient_buffers_state, free_double_gradient_buffers_state, apply_gradients, free_base_opt};

optimizer* cnstr_adamax_optimizer(const double beta1, const double beta2) {
    optimizer* opt = malloc(sizeof(optimizer));
    double* d = malloc(sizeof(double) * 2);
    d[0] = beta1;
    d[1] = beta2;

    opt->params = d;
    opt->vtable = &adamax_vtable;

    return opt;
}