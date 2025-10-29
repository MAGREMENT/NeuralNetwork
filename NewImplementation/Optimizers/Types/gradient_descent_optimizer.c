//
// Created by zacha on 23-10-25.
//

#include "gradient_descent_optimizer.h"

#include <stdlib.h>

static void apply_gradients(const optimizer* opt, double* to, const double* gradients, const int count, const optimizer_args args) {
    for (int i = 0; i < count; i++) {
        to[i] -= gradients[i] * args.learning_rate;
    }
}

static void free_gdo(optimizer* opt) {
    free(opt);
}

optimizer_vtable gdo_vtable = {cnstr_empty_state, free_empty_state, apply_gradients, free_gdo};

inline optimizer* cnstr_gradient_descent_optimizer() {
    optimizer* opt = malloc(sizeof(optimizer));
    opt->vtable = &gdo_vtable;
    return opt;
}
