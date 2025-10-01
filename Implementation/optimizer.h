//
// Created by zacha on 01-10-25.
//

#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "neural_network.h"

typedef struct optimizer optimizer;

struct optimizer {
    void* params;
    void* (*createState);
    void (*apply_gradients)(optimizer* opt, void* state, layer to, layer_data gradients, double learningRate, void* optimizerState);
    void (*free_state)(void* state);
    void (*free)(optimizer* optimizer);
};

#endif //OPTIMIZER_H
