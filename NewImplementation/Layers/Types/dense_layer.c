//
// Created by zacha on 20-10-25.
//

#include "dense_layer.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../../multi-threading.h"
#include "../../Util/rand_util.h"

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

typedef struct parallel_dense_forward_params {
    mt_dense_layer_params* params;
    const double* inputs;
    double* outputs;
    int in_count;
    int out_count;
} parallel_dense_forward_params;

static unsigned long async_dense_forward(void* params) {
    const parallel_range_data* data = params;
    parallel_dense_forward_params* p = data->params;

    for(int o = data->range.from; o < data->range.to; o++){
        double n = p->params->biases[o];

        for(int i = 0; i < p->in_count; i++){
            const int ind = i * p->out_count + o;
            n += p->inputs[i] * p->params->weights[ind];
        }

        p->outputs[o] = n;
    }

    return 0;
}

static void mt_dense_forward(const layer* l, const double* inputs, double* outputs) {
    mt_dense_layer_params* p = l->params;

    parallel_dense_forward_params* pdfp = malloc(sizeof(parallel_dense_forward_params));
    *pdfp = (parallel_dense_forward_params) {p, inputs, outputs, l->in_count, l->out_count};

    exec_range_parallel(async_dense_forward, pdfp, l->out_count, p->threadCount);

    free(pdfp);
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
            gradients[i * l->out_count + o] += deltas[o] * inputs[i];
        }

        const int ind = l->out_count * l->in_count + o;
        gradients[ind] += deltas[o];
    }
}

static void apply_gradients_to_dense(const layer* l, const double* gradients, const optimizer* opt, optimizer_args args) {
  	const dense_layer_params* p = l->params;

    opt->vtable->apply_gradients(opt, p->weights, gradients, l->in_count * l->out_count, args);
    opt->vtable->apply_gradients(opt, p->biases, gradients + l->in_count * l->out_count, l->out_count, args);
}

static void dense_export(const layer* l, double* parameters) {
    const dense_layer_params* p = l->params;

    memcpy(parameters, p->weights, sizeof(double) * l->in_count * l->out_count);
    memcpy(parameters + l->in_count * l->out_count, p->biases, sizeof(double) * l->out_count);
}

static void dense_import(const layer* l, const double* parameters) {
    const dense_layer_params* p = l->params;

    memcpy(p->weights, parameters, sizeof(double) * l->in_count * l->out_count);
    memcpy(p->biases, parameters + l->in_count * l->out_count, sizeof(double) * l->out_count);
}

layer_vtable dense_vtable = {dense_forward, dense_backward, dense_delta_to_gradients, apply_gradients_to_dense, dense_export, dense_import, free_dense_layer};

layer* cnstr_dense_layer(const int inputCount, const int outputCount, void (*initialize)(const layer* l)) {
    layer* l = malloc(sizeof(layer));
    dense_layer_params* p = malloc(sizeof(dense_layer_params));

    p->weights = malloc(sizeof(double) * inputCount * outputCount);
    p->biases = malloc(sizeof(double) * outputCount);

    l->params = p;
    l->in_count = inputCount;
    l->out_count = outputCount;
    l->gradient_count = inputCount * outputCount + outputCount;

    l->initialize = initialize;
    l->vtable = &dense_vtable;

    return l;
}

layer_vtable mt_dense_vtable = {mt_dense_forward, dense_backward, dense_delta_to_gradients, apply_gradients_to_dense, dense_export, dense_import, free_dense_layer};

//TODO
layer* cnstr_multi_thread_dense_layer(const int inputCount, const int outputCount, const int thread_count, void (*initialize)(const layer* l)) {
    layer* l = malloc(sizeof(layer));
    mt_dense_layer_params* p = malloc(sizeof(mt_dense_layer_params));

    p->weights = malloc(sizeof(double) * inputCount * outputCount);
    p->biases = malloc(sizeof(double) * outputCount);
    p->threadCount = thread_count;

    l->params = p;
    l->in_count = inputCount;
    l->out_count = outputCount;
    l->gradient_count = inputCount * outputCount + outputCount;

    l->initialize = initialize;
    l->vtable = &mt_dense_vtable;

    return l;
}

inline void set_weights(const layer* l, double values[]) {
    const dense_layer_params* p = l->params;
    memcpy(p->weights, values, sizeof(double) * l->in_count * l->out_count);
}

inline void set_biases(const layer* l, double values[]) {
    const dense_layer_params* p = l->params;
    memcpy(p->biases, values, sizeof(double) * l->out_count);
}

void set_all_weights_and_biases(const layer* l, const double weights, const double biases) {
    const dense_layer_params* p = l->params;

    for(int o = 0; o < l->out_count; o++) {
        for(int i = 0; i < l->in_count; i++) {
            p->weights[i * l->out_count + o] = weights;
        }

        p->biases[o] = biases;
    }
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

inline void initialize_dense_to_two(const layer* l) {
    init_d_to_d(l, 2);
}

static void biasesToZero(const dense_layer_params* p, const int count) {
    for (int i = 0; i < count; i++) {
        p->biases[i] = 0;
    }
}

void initialize_dense_random(const layer* layer) {
    const int total = layer->in_count * layer->out_count;
    const dense_layer_params* p = layer->params;

    for (int i = 0; i < total; i++) {
        p->weights[i] = rand_d_std_nrml_distr() * 0.01;
    }

    biasesToZero(p, layer->out_count);
}

void initialize_dense_he(const layer* layer) {
    const int total = layer->in_count * layer->out_count;
    const dense_layer_params* p = layer->params;

    const double scale = sqrt(2.0 / layer->in_count);
    for (int i = 0; i < total; i++) {
        p->weights[i] = rand_d_std_nrml_distr() * scale;
    }

    biasesToZero(p, layer->out_count);
}

void initialize_dense_xavier(const layer* layer) {
    const int total = layer->in_count * layer->out_count;
    const dense_layer_params* p = layer->params;

    const double scale = sqrt(1.0 / layer->in_count);
    for (int i = 0; i < total; i++) {
        p->weights[i] = rand_d_std_nrml_distr() * scale;
    }

    biasesToZero(p, layer->out_count);
}


