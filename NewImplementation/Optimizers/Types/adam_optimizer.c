//
// Created by zacha on 29-10-25.
//

#include "adam_optimizer.h"

#include <stdlib.h>
#include <math.h>

#define EPSILON 1e-8

static void apply_gradients(const optimizer* opt, double* to, const double* gradients, const int count, const optimizer_args args) {
    double* v1 = ((double***)args.state)[0][args.layerIndex];
    double* v2 = ((double***)args.state)[1][args.layerIndex];
    const double beta1 = ((double*)opt->params)[0];
    const double beta2 = ((double*)opt->params)[1];

    for (int i = 0; i < count; i++) {
        const double grad = gradients[i];
        v1[i] = beta1 * v1[i] + (1 - beta1) * grad;
        v2[i] = beta2 * v2[i] + (1 - beta2) * grad * grad;

        const double m = v1[i] / (1 - pow(beta1, args.iteration));
        const double v = v2[i] / (1 - pow(beta2, args.iteration));

        to[i] -= args.learning_rate * m / sqrt(v + EPSILON);
    }
}

optimizer_vtable adam_vtable = {cnstr_double_gradient_buffers_state, free_double_gradient_buffers_state, apply_gradients, free_base_opt};

optimizer* cnstr_adam_optimizer(const double beta1, const double beta2) {
    optimizer* opt = malloc(sizeof(optimizer));
    double* d = malloc(sizeof(double) * 2);
    d[0] = beta1;
    d[1] = beta2;

    opt->params = d;
    opt->vtable = &adam_vtable;

    return opt;
}