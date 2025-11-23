//
// Created by zacha on 29-10-25.
//

#include "adam_optimizer.h"
#include "../../Util/range.h"

#include <stdlib.h>
#include <math.h>

#define EPSILON 1e-8

static void apply_gradients(const optimizer* opt, double* to, const double* gradients, const range r,
        const double beta1, const double beta2, const optimizer_args args) {
    double* v1 = ((double***)args.state)[0][args.layerIndex];
    double* v2 = ((double***)args.state)[1][args.layerIndex];
    const double beta1t = 1 - pow(beta1, args.iteration);
    const double beta2t = 1 - pow(beta2, args.iteration);

    for (int i = r.from; i < r.to; i++) {
        const double grad = gradients[i];
        v1[i] = beta1 * v1[i] + (1 - beta1) * grad;
        v2[i] = beta2 * v2[i] + (1 - beta2) * grad * grad;

        const double m = v1[i] / beta1t;
        const double v = v2[i] / beta2t;

        to[i] -= args.learning_rate * m / sqrt(v + EPSILON);
    }
}

static void st_apply_gradients(const optimizer* opt, double* to, const double* gradients, const int count, const optimizer_args args) {
    apply_gradients(opt, to, gradients, (range){0, count},
        ((double*)opt->params)[0], ((double*)opt->params)[1], args);
}

optimizer_vtable adam_vtable = {cnstr_double_gradient_buffers_state, free_double_gradient_buffers_state, st_apply_gradients, free_base_opt};

optimizer* cnstr_adam_optimizer(const double beta1, const double beta2) {
    optimizer* opt = malloc(sizeof(optimizer));
    double* d = malloc(sizeof(double) * 2);
    d[0] = beta1;
    d[1] = beta2;

    opt->params = d;
    opt->vtable = &adam_vtable;

    return opt;
}

typedef struct mt_adam_data {
    double beta1;
    double beta2;
    parallel_range_executor* executor;
} mt_adam_data;

void free_mt_adam(optimizer* opt) {
    free_pr_executor(((mt_adam_data*)opt->params)->executor);
    free_base_opt(opt);
}

typedef struct mt_adam_ag_params {
    const optimizer* opt;
    double* to;
    const double* gradients;
    const double beta1;
    const double beta2;
    const optimizer_args args;
} mt_adam_ag_params;

unsigned long async_apply_gradients(void* params) {
    const parallel_range_data* data = params;
    const mt_adam_ag_params* ag_params = data->params;

    apply_gradients(ag_params->opt, ag_params->to, ag_params->gradients, to_range(data->range), ag_params->beta1,
        ag_params->beta2, ag_params->args);

    return 0;
}

static void mt_apply_gradients(const optimizer* opt, double* to, const double* gradients, const int count, const optimizer_args args) {
    const mt_adam_data* data = opt->params;
    mt_adam_ag_params ag_params = {opt, to, gradients, data->beta1, data->beta2, args};
    exec_parallel_range(data->executor, async_apply_gradients, &ag_params, (range){0, count});
}

optimizer_vtable mt_adam_vtable = {cnstr_double_gradient_buffers_state, free_double_gradient_buffers_state, mt_apply_gradients, free_mt_adam};

optimizer* cnstr_mt_adam_optimizer(const double beta1, const double beta2, thread_pool* pool, const int parallelCount) {
    optimizer* opt = malloc(sizeof(optimizer));
    mt_adam_data* d = malloc(sizeof(mt_adam_data));
    d->beta1 = beta1;
    d->beta2 = beta2;
    d->executor = alloc_pr_executor(pool, parallelCount);

    opt->params = d;
    opt->vtable = &mt_adam_vtable;

    return opt;
}