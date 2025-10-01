//
// Created by zacha on 01-10-25.
//

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "layer.h"

typedef struct optimizer optimizer;

struct optimizer {
    void* params;
    void* (*create_state)(optimizer* opt);
    void (*apply_gradients)(optimizer* opt, void* state, layer to, layer_data gradients, double learningRate);
    void (*free_state)(void* state);
    void (*free)(optimizer* optimizer);
};

optimizer* create_gradient_descent_optimizer();

#endif //OPTIMIZER_H
