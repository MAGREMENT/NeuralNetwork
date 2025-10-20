//
// Created by zacha on 20-10-25.
//

#include "dense_layer.h"

#include <stdlib.h>

typedef struct dense_layer_params dense_layer_params;

struct dense_layer_params {
    int in_count;
    int out_count;
    double* weights;
    double* biases;
};

static void free_dense_layer(layer* l) {
    dense_layer_params* p = l->params;
    free(p->weights);
    free(p->biases);
    free(p);
    free(l);
}

static double* dense_forward(const layer* l, double* inputs, int* didAllocate) {
    const dense_layer_params* p = l->params;

    double* result = malloc(p->out_count * sizeof(double));

    for(int o = 0; o < p->out_count; o++){
        double n = p->biases[o];

        for(int i = 0; i < p->in_count; i++){
            const int ind = i * p->out_count + o;
            n += inputs[i] * p->weights[ind];
        }

        result[o] = n;
    }

    *didAllocate = true;
    return result;
}

inline layer* cnstr_dense_layer(const int inputCount, const int outputCount, void (*initialize)(layer* l)) {
    layer* l = malloc(sizeof(layer));
    dense_layer_params* p = malloc(sizeof(dense_layer_params));
    p->weights = malloc(sizeof(double) * inputCount * outputCount);
    p->biases = malloc(sizeof(double) * outputCount);
    p->in_count = inputCount;
    p->out_count = outputCount;

    l->params = p;
    l->functions.initialize = initialize;
    l->functions.forward = dense_forward;
    l->functions.free = free_dense_layer;

    return l;
}

static void init_d_to_d(const layer* l, const double d) {
    const dense_layer_params* p = l->params;

    for(int o = 0; o < p->out_count; o++){
        p->biases[o] = d;

        for(int i = 0; i < p->in_count; i++){
            const int ind = i * p->out_count + o;
            p->weights[ind] = d;
        }
    }
}

inline void initialize_dense_to_zero(layer* l) {
    init_d_to_d(l, 0);
}

inline void initialize_dense_to_one(layer* l) {
    init_d_to_d(l, 1);
}
