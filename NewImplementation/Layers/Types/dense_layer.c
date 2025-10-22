//
// Created by zacha on 20-10-25.
//

#include "dense_layer.h"

#include <stdlib.h>

typedef struct dense_layer_params dense_layer_params;

struct dense_layer_params {
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

static void dense_forward(const layer* l, const double* inputs, double* outputs) {
    const dense_layer_params* p = l->params;

    for(int o = 0; o < l->out_count; o++){
        double n = p->biases[o];

        for(int i = 0; i < l->in_count; i++){
            const int ind = i * l->out_count + o;
            n += inputs[i] * p->weights[ind];
        }

        outputs[o] = n;
    }
}

static void dense_backward(const layer* l, const double* inputs, const double* deltas, double* outputs) {
    const dense_layer_params* p = l->params;

    for(int i = 0; i < l->in_count; i++) {
        double value = 0;
        for(int o = 0; o < l->out_count; o++) {
            const double w = p->weights[i * l->out_count + o];
            const double nv = deltas[o];
            value += nv * w;
        }

        outputs[i] = value;
    }
}

static void dense_delta_to_gradients(const layer* l, const double* inputs, const double* deltas, double* gradients) {
    for(int o = 0; o < l->out_count; o++) {
        for (int i = 0; i < l->in_count; i++) {
            gradients[i * l->out_count + o] = deltas[o] * inputs[i];
        }

        gradients[l->out_count * l->in_count + o] = deltas[o];
    }
}

static void apply_gradients_to_dense(const layer* l, const double* gradients, const optimizer* opt, optimizer_args args) {
  	const dense_layer_params* p = l->params;

    opt->apply_gradients(p->weights, gradients, l->in_count * l->out_count, args);
    opt->apply_gradients(p->biases, gradients + l->in_count * l->out_count, l->out_count, args);
}

inline layer* cnstr_dense_layer(const int inputCount, const int outputCount, void (*initialize)(const layer* l)) {
    layer* l = malloc(sizeof(layer));
    dense_layer_params* p = malloc(sizeof(dense_layer_params));
    p->weights = malloc(sizeof(double) * inputCount * outputCount);
    p->biases = malloc(sizeof(double) * outputCount);

    l->params = p;
    l->in_count = inputCount;
    l->out_count = outputCount;
    l->gradient_count = inputCount * outputCount + outputCount;

    l->functions.initialize = initialize;
    l->functions.forward = dense_forward;
    l->functions.backward = dense_backward;
    l->functions.free = free_dense_layer;
    l->functions.deltas_to_gradients = dense_delta_to_gradients;
    l->functions.apply_gradients = apply_gradients_to_dense;

    return l;
}

static void init_d_to_d(const layer* l, const double d) {
    const dense_layer_params* p = l->params;

    for(int o = 0; o < l->out_count; o++){
        p->biases[o] = d;

        for(int i = 0; i < l->in_count; i++){
            const int ind = i * l->out_count + o;
            p->weights[ind] = d;
        }
    }
}

inline void initialize_dense_to_zero(const layer* l) {
    init_d_to_d(l, 0);
}

inline void initialize_dense_to_one(const layer* l) {
    init_d_to_d(l, 1);
}
