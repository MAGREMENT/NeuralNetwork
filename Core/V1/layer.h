//
// Created by zacha on 01-10-25.
//

#ifndef OLD_LAYER_H
#define OLD_LAYER_H

typedef struct layer layer;

struct layer {
    int in_count;
    int out_count;
    double* weights;
    double* biases;

    void (*initialization)(layer* layer);
    double (*activation)(double, void*);
    double (*activationDerivative)(double, void*);
    void* (*processInputs)(double*, int);
    void (*freeData)(void*);
};

typedef struct layer_data {
    double* weights;
    double* biases;
} layer_data;

layer_data* alloc_layer_data_array(layer* layers, int layerCount, int copyValues);
void free_layer_data_array(layer_data* layers, int count);

#endif
