//
// Created by zacha on 20-10-25.
//

#include "dense_layer.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../../multi-threading.h"
#include "../../Util/rand_util.h"

static double* get_biases(const layer* l) {
    return l->parameters + l->in_count * l->out_count;
}

static void dense_forward(const layer* l, const double* inputs, double* outputs) {
    const double* biases = get_biases(l);

    for(int o = 0; o < l->out_count; o++){
        double n = biases[o];

        for(int i = 0; i < l->in_count; i++){
            const int ind = i * l->out_count + o;
            n += inputs[i] * l->parameters[ind];
        }

        outputs[o] = n;
    }
}

typedef struct parallel_dense_forward_params {
    const layer* layer;
    const double* inputs;
    double* outputs;
} parallel_dense_forward_params;

static unsigned long async_dense_forward(void* params) {
    const parallel_range_data* data = params;
    const parallel_dense_forward_params* p = data->params;

    const double* biases = get_biases(p->layer);

    for(int o = data->range.from; o < data->range.to; o++){
        double n = biases[o];

        for(int i = 0; i < p->layer->in_count; i++){
            const int ind = i * p->layer->out_count + o;
            n += p->inputs[i] * p->layer->parameters[ind];
        }

        p->outputs[o] = n;
    }

    return 0;
}

static void mt_dense_forward(const layer* l, const double* inputs, double* outputs) {
    const int threadCount = *(int*)l->data;
    parallel_dense_forward_params pdfp = (parallel_dense_forward_params) {l, inputs, outputs};
    exec_range_parallel(async_dense_forward, &pdfp, l->out_count, threadCount);
}

static void dense_backward(const layer* l, const double* inputs, const double* deltas, double* outputs) {
    for(int i = 0; i < l->in_count; i++) {
        double value = 0;
        for(int o = 0; o < l->out_count; o++) {
            const double w = l->parameters[i * l->out_count + o];
            const double nv = deltas[o];
            value += nv * w;
        }

        outputs[i] = value;
    }
}

typedef struct parallel_dense_backward_params {
    const layer* layer;
    const double* deltas;
    double* outputs;
} parallel_dense_backward_params;

static unsigned long async_dense_backward(void* params) {
    const parallel_range_data* data = params;
    const parallel_dense_backward_params* p = data->params;

    for(int i = data->range.from; i < data->range.to; i++) {
        double value = 0;
        for(int o = 0; o < p->layer->out_count; o++) {
            const double w = p->layer->parameters[i * p->layer->out_count + o];
            const double nv = p->deltas[o];
            value += nv * w;
        }

        p->outputs[i] = value;
    }

    return 0;
}

static void mt_dense_backward(const layer* l, const double* inputs, const double* deltas, double* outputs) {
    const int threadCount = *(int*)l->data;
    parallel_dense_backward_params pdfp = (parallel_dense_backward_params) {l, deltas, outputs};
    exec_range_parallel(async_dense_backward, &pdfp, l->in_count, threadCount);
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

typedef struct parallel_dense_dtg_params {
    const layer* layer;
    const double* inputs;
    const double* deltas;
    double* outputs;
} parallel_dense_dtg_params;

static unsigned long async_dense_delta_to_gradients(void* params) {
    const parallel_range_data* data = params;
    const parallel_dense_dtg_params* p = data->params;

    for(int o = data->range.from; o < data->range.to; o++) {
        for (int i = 0; i < p->layer->in_count; i++) {
            p->outputs[i * p->layer->out_count + o] += p->deltas[o] * p->inputs[i];
        }

        const int ind = p->layer->out_count * p->layer->in_count + o;
        p->outputs[ind] += p->deltas[o];
    }
    return 0;
}

static void mt_dense_delta_to_gradients(const layer* l, const double* inputs, const double* deltas, double* gradients) {
    const int threadCount = *(int*)l->data;
    parallel_dense_dtg_params pdfp = (parallel_dense_dtg_params) {l, inputs, deltas, gradients};
    exec_range_parallel(async_dense_delta_to_gradients, &pdfp, l->out_count, threadCount);
}

layer_vtable dense_vtable = {NULL, dense_forward, NULL, dense_backward, dense_delta_to_gradients, default_layer_free};

layer* cnstr_dense_layer(const int inputCount, const int outputCount, void (*initialize)(const layer* l)) {
    layer* l = malloc(sizeof(layer));

    l->data = NULL;
    l->in_count = inputCount;
    l->out_count = outputCount;

    l->parameters_count = inputCount * outputCount + outputCount;
    l->parameters = malloc(sizeof(double) * l->parameters_count);

    l->initialize = initialize;
    l->vtable = &dense_vtable;

    return l;
}

layer_vtable mt_dense_vtable = {NULL, mt_dense_forward, NULL, mt_dense_backward, mt_dense_delta_to_gradients, default_layer_free};

layer* cnstr_multi_thread_dense_layer(const int inputCount, const int outputCount, const int thread_count, void (*initialize)(const layer* l)) {
    layer* l = malloc(sizeof(layer));
    int* tc = malloc(sizeof(int));
    *tc = thread_count;

    l->data = tc;

    l->in_count = inputCount;
    l->out_count = outputCount;

    l->parameters_count = inputCount * outputCount + outputCount;
    l->parameters = malloc(sizeof(double) * l->parameters_count);

    l->initialize = initialize;
    l->vtable = &mt_dense_vtable;

    return l;
}

inline void set_weights(const layer* l, double values[]) {
    memcpy(l->parameters, values, sizeof(double) * l->in_count * l->out_count);
}

inline void set_biases(const layer* l, double values[]) {
    memcpy(get_biases(l), values, sizeof(double) * l->out_count);
}

void set_all_weights_and_biases(const layer* l, const double weights, const double biases) {
    double* b = get_biases(l);

    for(int o = 0; o < l->out_count; o++) {
        for(int i = 0; i < l->in_count; i++) {
            l->parameters[i * l->out_count + o] = weights;
        }

        b[o] = biases;
    }
}

static void init_d_to_d(const layer* l, const double d) {
    double* b = get_biases(l);

    for(int o = 0; o < l->out_count; o++){
        b[o] = d;

        for(int i = 0; i < l->in_count; i++){
            const int ind = i * l->out_count + o;
            l->parameters[ind] = d;
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

void initialize_dense_random(const layer* layer) {
    const int total = layer->in_count * layer->out_count;

    for (int i = 0; i < total; i++) {
        layer->parameters[i] = rand_d_std_nrml_distr() * 0.01;
    }

    memset(get_biases(layer), 0, sizeof(double) * layer->out_count);
}

void initialize_dense_he(const layer* layer) {
    const int total = layer->in_count * layer->out_count;

    const double scale = sqrt(2.0 / layer->in_count);
    for (int i = 0; i < total; i++) {
        layer->parameters[i] = rand_d_std_nrml_distr() * scale;
    }

    memset(get_biases(layer), 0, sizeof(double) * layer->out_count);
}

void initialize_dense_xavier(const layer* layer) {
    const int total = layer->in_count * layer->out_count;

    const double scale = sqrt(1.0 / layer->in_count);
    for (int i = 0; i < total; i++) {
        layer->parameters[i] = rand_d_std_nrml_distr() * scale;
    }

    memset(get_biases(layer), 0, sizeof(double) * layer->out_count);
}


