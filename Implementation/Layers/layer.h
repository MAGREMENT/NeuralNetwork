//
// Created by zacha on 20-10-25.
//

#ifndef LAYER_H
#define LAYER_H

typedef struct layer layer;

typedef struct layer_functions {
    double* (*forward)(const layer* l, double* inputs, int* didAllocate);
    double* (*backward)(const layer* l, double* inputs, double* deltas);
    void (*initialize)(layer* l);
    void (*free)(layer* l);
} layer_functions;

struct layer {
    void* params;

    layer_functions functions;
};

void no_initialization(layer* l);
void default_layer_free(layer* l);

#endif //LAYER_H
