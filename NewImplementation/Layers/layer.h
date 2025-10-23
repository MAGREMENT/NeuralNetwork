//
// Created by zacha on 20-10-25.
//

#ifndef LAYER_H
#define LAYER_H

#include "../Optimizers/optimizer.h"

enum layer_types {
    DENSE,
    ACTIVATION
};

typedef struct layer layer;

typedef struct layer_functions {
    void (*forward)(const layer* l, const double* inputs, double* outputs);
    void (*backward)(const layer* l, const double* inputs, const double* deltas, double* outputs);
    void (*deltas_to_gradients)(const layer* l, const double* inputs, const double* deltas, double* gradients);
    void (*apply_gradients)(const layer* l, const double* gradients, const optimizer* opt, optimizer_args args);
    void (*initialize)(const layer* l);
    void (*free)(layer* l);
} layer_functions;

struct layer {
    void* params;
    int in_count;
    int out_count;
    int gradient_count;

    //TODO to vtable
    layer_functions functions;
};

void no_initialization(const layer* l);
void default_layer_free(layer* l);

#endif //LAYER_H
