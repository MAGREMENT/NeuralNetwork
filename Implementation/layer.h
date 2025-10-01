//
// Created by zacha on 01-10-25.
//

#ifndef LAYER_H
#define LAYER_H

typedef struct layer {
    int in_count;
    int out_count;
    double* weights;
    double* biases;

    double (*activation)(double, void*);
    double (*activationDerivative)(double, void*);
    void* (*processInputs)(double*, int);
    void (*freeData)(void*);
} layer;

typedef struct layer_data {
    double* weights;
    double* biases;
} layer_data;

void free_layer_data_array(layer_data* gradients, int count);

#endif //LAYER_H
