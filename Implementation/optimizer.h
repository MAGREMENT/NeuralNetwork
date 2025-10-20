//
// Created by zacha on 01-10-25.
//

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "old_layer.h"
#include "list.h"

typedef struct optimizer optimizer;

struct optimizer {
    void* params;
    void* (*create_state)(optimizer* opt, old_layer* layers, int layerCount);
    void (*apply_gradients)(optimizer* opt, void* state, old_layer* layers, layer_data* gradients, int layerCount,
        int iteration, double learningRate);
    void (*free_state)(void* state, int layerCount);
    void (*free)(optimizer* optimizer);
    s_arr* (*alloc_to_hyper)(optimizer* opt, int startIndentation); //TODO finish implementation
};

optimizer* create_gradient_descent_optimizer();
optimizer* create_momentum_gradient_descent_optimizer(double momentum);
optimizer* create_nesterov_optimizer(double momentum);
optimizer* create_rmsprop_optimizer(double decay);
optimizer* create_adam_optimizer(double beta1, double beta2);

#endif //OPTIMIZER_H
