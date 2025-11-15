//
// Created by zacha on 15-11-25.
//

#include "adamw_optimizer.h"

#include <math.h>
#include <stdlib.h>

#define EPSILON 1e-8

static void apply_gradients(const optimizer* opt, double* to, const double* gradients, const int count, const optimizer_args args) {
    double* v1 = ((double***)args.state)[0][args.layerIndex];
    double* v2 = ((double***)args.state)[1][args.layerIndex];
    const double beta1 = ((double*)opt->params)[0];
    const double beta2 = ((double*)opt->params)[1];
    const double decay = ((double*)opt->params)[2];
    const double beta1t = 1 - pow(beta1, args.iteration);
    const double beta2t = 1 - pow(beta2, args.iteration);

    for (int i = 0; i < count; i++) {
        const double grad = gradients[i];
        v1[i] = beta1 * v1[i] + (1 - beta1) * grad;
        v2[i] = beta2 * v2[i] + (1 - beta2) * grad * grad;

        const double m = v1[i] / beta1t;
        const double v = v2[i] / beta2t;

        const double before = to[i];
        to[i] -= args.learning_rate * m / sqrt(v + EPSILON);
        to[i] -= decay * before * args.learning_rate;
    }
}

optimizer_vtable adamw_vtable = {cnstr_double_gradient_buffers_state, free_double_gradient_buffers_state, apply_gradients, free_base_opt};

optimizer* cnstr_adamw_optimizer(const double beta1, const double beta2, const double decay) {
    optimizer* opt = malloc(sizeof(optimizer));
    double* d = malloc(sizeof(double) * 3);
    d[0] = beta1;
    d[1] = beta2;
    d[2] = decay;

    opt->params = d;
    opt->vtable = &adamw_vtable;

    return opt;
}