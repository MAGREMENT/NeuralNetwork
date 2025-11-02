//
// Created by zacha on 20-10-25.
//

#ifndef NEWIMPLEMENTATION_LAYER_H
#define NEWIMPLEMENTATION_LAYER_H

#include "../Optimizers/optimizer.h"

enum layer_types {
    DENSE,
    ACTIVATION
};

typedef struct layer layer;

typedef struct layer_vtable {
    void (*forward)(const layer* l, const double* inputs, double* outputs);
    void (*backward)(const layer* l, const double* inputs, const double* deltas, double* outputs);
    void (*deltas_to_gradients)(const layer* l, const double* inputs, const double* deltas, double* gradients);
    void (*apply_gradients)(const layer* l, const double* gradients, const optimizer* opt, optimizer_args args);
    void (*free)(layer* l);
} layer_vtable;

struct layer {
    void* params;
    int in_count;
    int out_count;
    int gradient_count;

    layer_vtable* vtable;
    void (*initialize)(const layer* l);
};

void no_initialization(const layer* l);
void default_layer_free(layer* l);

#endif //NEWIMPLEMENTATION_LAYER_H
