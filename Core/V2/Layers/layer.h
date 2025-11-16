//
// Created by zacha on 20-10-25.
//

#ifndef NEWIMPLEMENTATION_LAYER_H
#define NEWIMPLEMENTATION_LAYER_H

#include "../Optimizers/optimizer.h"

enum layer_types {
    DENSE,
    ACTIVATION,
    CONVOLUTIONAL,
    POOLING
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
    void* data;

    int in_count;
    int out_count;

    double* parameters;
    int parameters_count;

    layer_vtable* vtable;
    void (*initialize)(const layer* l);
};

extern void no_initialization(const layer* l);
extern void default_layer_free(layer* l);

extern void no_delta_to_gradients(const layer* l, const double* inputs, const double* deltas, double* gradients);
extern void apply_no_gradients(const layer* l, const double* gradients, const optimizer* opt, optimizer_args args);

#endif //NEWIMPLEMENTATION_LAYER_H
