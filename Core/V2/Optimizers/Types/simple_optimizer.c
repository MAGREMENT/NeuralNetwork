//
// Created by zacha on 23-10-25.
//

#include "simple_optimizer.h"

#include <stdlib.h>

static void apply_gradients(const optimizer* opt, double* to, const double* gradients, const int count, const optimizer_args args) {
    for (int i = 0; i < count; i++) {
        to[i] -= gradients[i] * args.learning_rate;
    }
}

optimizer_vtable s_vtable = {cnstr_empty_state, free_empty_state, apply_gradients, free_empty_opt};

optimizer* cnstr_simple_optimizer() {
    optimizer* opt = malloc(sizeof(optimizer));
    opt->params = NULL;
    opt->vtable = &s_vtable;
    return opt;
}
