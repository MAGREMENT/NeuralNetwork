//
// Created by zacha on 16-11-25.
//

#include "dropout_layer.h"

#include <stdlib.h>

#include "../../Util/rand_util.h"
#include "../../Util/Collections/bitset.h"

typedef struct dropout_layer_data {
    double rate;
    int* bitset;
} dropout_layer_data;

static void free_dropout_layer(layer* l) {
    const dropout_layer_data* d = l->data;
    free(d->bitset);
    default_layer_free(l);
}

static void on_dropout_predict_start(const layer* l) {
    const dropout_layer_data* d = l->data;
    full_set(d->bitset, l->out_count);
}

static void on_dropout_learn_start(const layer* l) {
    const dropout_layer_data* d = l->data;

    for (int i = 0; i < l->out_count; i++) {
        if (rand_d(0, 1) > d->rate) set(d->bitset, i);
        else unset(d->bitset, i);
    }
}

static void inverted_dropout_pass(const layer* l, const double* inputs, double* outputs) {
    const dropout_layer_data* d = l->data;
    const double scale = 1.0 / (1.0 - d->rate);

    for (int i = 0; i < l->out_count; i++) {
        outputs[i] = is_set(d->bitset, i) ? inputs[i] * scale : 0;
    }
}

static void inverted_dropout_forward(const layer* l, const double* inputs, double* outputs) {
    inverted_dropout_pass(l, inputs, outputs);
}

static void inverted_dropout_backward(const layer* l, const double* inputs, const double* deltas, double* outputs) {
    inverted_dropout_pass(l, deltas, outputs);
}

layer_vtable dropout_vtables[] = {
    {on_dropout_predict_start, inverted_dropout_forward, on_dropout_learn_start, inverted_dropout_backward, no_delta_to_gradients, NULL, free_dropout_layer}
};

layer* cnstr_dropout_layer(const int type, const int outCount, const double rate) {
    layer* l = malloc(sizeof(layer));
    dropout_layer_data* data = malloc(sizeof(dropout_layer_data));

    data->rate = rate;
    data->bitset = alloc_bitset(outCount);

    l->data = data;
    l->in_count = outCount;
    l->out_count = outCount;

    l->parameters_count = 0;
    l->parameters = NULL;

    l->vtable = dropout_vtables + type;
    l->initialize = no_initialization;

    return l;
}
