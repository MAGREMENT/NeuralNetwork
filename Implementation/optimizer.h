//
// Created by zacha on 01-10-25.
//

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "layer.h"

typedef struct optimizer optimizer;

struct optimizer {
    void* params;
    void* (*create_state)(optimizer* opt, layer* layers, int layerCount);
    void (*apply_gradients)(optimizer* opt, void* state, layer* layers, layer_data* gradients, int layerCount,
        int iteration, double learningRate);
    void (*free_state)(void* state, int layerCount);
    void (*free)(optimizer* optimizer);
};

optimizer* create_gradient_descent_optimizer();
optimizer* create_momentum_gradient_descent_optimizer(double momentum);
optimizer* create_nesterov_optimizer(double momentum);
optimizer* create_rmsprop_optimizer(double decay);
optimizer* create_adam_optimizer(double beta1, double beta2);

#endif //OPTIMIZER_H
