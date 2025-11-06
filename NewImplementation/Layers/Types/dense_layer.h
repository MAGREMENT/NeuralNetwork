//
// Created by zacha on 20-10-25.
//

#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "../layer.h"

typedef struct dense_layer_params {
    double* weights;
    double* biases;
} dense_layer_params;

typedef struct mt_dense_layer_params {
    double* weights;
    double* biases;
    int threadCount;
} mt_dense_layer_params;

layer* cnstr_dense_layer(int inputCount, int outputCount, void (*initialize)(const layer* l));
layer* cnstr_multi_thread_dense_layer(int inputCount, int outputCount, int thread_count, void (*initialize)(const layer* l));

void set_weights(const layer* l, double values[]);
void set_biases(const layer* l, double values[]);
void set_all_weights_and_biases(const layer* l, double weights, double biases);

void initialize_dense_to_zero(const layer* l);
void initialize_dense_to_one(const layer* l);
void initialize_dense_to_two(const layer* l);
void initialize_dense_random(const layer* layer);
void initialize_dense_he(const layer* layer);
void initialize_dense_xavier(const layer* layer);

#endif //DENSE_LAYER_H
