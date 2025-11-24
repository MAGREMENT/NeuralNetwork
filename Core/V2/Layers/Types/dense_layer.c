//
// Created by zacha on 20-10-25.
//

#include "dense_layer.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>


#include "../../Util/rand_util.h"

static double* get_biases(const layer* l) {
    return l->parameters + l->in_count * l->out_count;
}

static void dense_forward(const layer* l, const double* inputs, double* outputs, const range r) {
    const double* biases = get_biases(l);

    for(int o = r.from; o < r.to; o++){
        double n = biases[o];

        for(int i = 0; i < l->in_count; i++){
            const int ind = i * l->out_count + o;
            n += inputs[i] * l->parameters[ind];
        }

        outputs[o] = n;
    }
}

static void st_dense_forward(const layer* l, const double* inputs, double* outputs) {
    dense_forward(l, inputs, outputs, (range){0, l->out_count});
}

typedef struct parallel_dense_forward_params {
    const layer* layer;
    const double* inputs;
    double* outputs;
} parallel_dense_forward_params;

static unsigned long async_dense_forward(void* params) {
    const parallel_range_data* data = params;
    const parallel_dense_forward_params* p = data->params;

    dense_forward(p->layer, p->inputs, p->outputs, to_range(data->range));

    return 0;
}

static void mt_dense_forward(const layer* l, const double* inputs, double* outputs) {
    parallel_dense_forward_params pdfp = (parallel_dense_forward_params) {l, inputs, outputs};
    exec_parallel_range(l->data, async_dense_forward, &pdfp, (range){0, l->out_count});
}

static void dense_backward(const layer* l, const double* deltas, double* outputs, const range r) {
    for(int i = r.from; i < r.to; i++) {
        double value = 0;
        for(int o = 0; o < l->out_count; o++) {
            const double w = l->parameters[i * l->out_count + o];
            const double nv = deltas[o];
            value += nv * w;
        }

        outputs[i] = value;
    }
}

static void st_dense_backward(const layer* l, const double* inputs, const double* deltas, double* outputs) {
    dense_backward(l, deltas, outputs, (range){0, l->in_count});
}

typedef struct parallel_dense_backward_params {
    const layer* layer;
    const double* deltas;
    double* outputs;
} parallel_dense_backward_params;

static unsigned long async_dense_backward(void* params) {
    const parallel_range_data* data = params;
    const parallel_dense_backward_params* p = data->params;

    dense_backward(p->layer, p->deltas, p->outputs, to_range(data->range));

    return 0;
}

static void mt_dense_backward(const layer* l, const double* inputs, const double* deltas, double* outputs) {
    parallel_dense_backward_params pdfp = (parallel_dense_backward_params) {l, deltas, outputs};
    exec_parallel_range(l->data, async_dense_backward, &pdfp, (range){0, l->in_count});
}

static void dense_delta_to_gradients(const layer* l, const double* inputs, const double* deltas, double* gradients, const range r) {
    for(int o = r.from; o < r.to; o++) {
        for (int i = 0; i < l->in_count; i++) {
            gradients[i * l->out_count + o] += deltas[o] * inputs[i];
        }

        const int ind = l->out_count * l->in_count + o;
        gradients[ind] += deltas[o];
    }
}

static void st_dense_delta_to_gradients(const layer* l, const double* inputs, const double* deltas, double* gradients) {
    dense_delta_to_gradients(l, inputs, deltas, gradients, (range){0, l->out_count});
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

    dense_delta_to_gradients(p->layer, p->inputs, p->deltas, p->outputs, to_range(data->range));

    return 0;
}

static void mt_dense_delta_to_gradients(const layer* l, const double* inputs, const double* deltas, double* gradients) {
    parallel_dense_dtg_params pdfp = (parallel_dense_dtg_params) {l, inputs, deltas, gradients};
    exec_parallel_range(l->data, async_dense_delta_to_gradients, &pdfp, (range){0, l->out_count});
}

layer_vtable dense_vtable = {NULL, st_dense_forward, NULL, st_dense_backward, st_dense_delta_to_gradients, empty_layer_free};

static layer* cnstr_base_dense_layer(const int inputCount, const int outputCount, void (*initialize)(const layer* l), layer_vtable* vtable) {
    layer* l = malloc(sizeof(layer));

    l->in_count = inputCount;
    l->out_count = outputCount;

    l->parameters_count = inputCount * outputCount + outputCount;
    l->parameters = malloc(sizeof(double) * l->parameters_count);

    l->initialize = initialize;
    l->vtable = vtable;

    return l;
}

layer* cnstr_dense_layer(const int inputCount, const int outputCount, void (*initialize)(const layer* l)) {
    layer* l = cnstr_base_dense_layer(inputCount, outputCount, initialize, &dense_vtable);
    l->data = NULL;
    return l;
}

layer_vtable wmt_dense_vtable = {NULL, mt_dense_forward, NULL, mt_dense_backward, mt_dense_delta_to_gradients, empty_layer_free};

layer* cnstr_multi_thread_dense_layer(const int inputCount, const int outputCount, thread_pool* pool, const int parallelCount, void (*initialize)(const layer* l)) {
    layer* l = cnstr_base_dense_layer(inputCount, outputCount, initialize, &wmt_dense_vtable);
    l->data = alloc_pr_executor(pool, parallelCount);
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


