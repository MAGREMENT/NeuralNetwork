//
// Created by zacha on 20-10-25.
//

#include "activation_layer.h"
#include "../../Util/math_util.h"

#include <math.h>
#include <stdlib.h>

#define LEAK 0.01

static void activation_delta_to_gradients(const layer* l, const double* inputs, const double* deltas, double* gradients) {

}

static void apply_gradients_to_activation(const layer* l, const double* gradients, const optimizer* opt, optimizer_args args) {

}

static void sigmoid_forward(const layer* l, const double* inputs, double* outputs) {
    for (int i = 0; i < l->out_count; i++) {
        outputs[i] = sigmoid(inputs[i]);
    }
}

static void sigmoid_backward(const layer* l, const double* inputs, const double* deltas, double* outputs) {
    for (int i = 0; i < l->out_count; i++) {
        const double a = sigmoid(inputs[i]);
        outputs[i] = deltas[i] * a * (1 - a);
    }
}

static void tanh_forward(const layer* l, const double* inputs, double* outputs) {
    for (int i = 0; i < l->out_count; i++) {
        const double e2 = exp(2 * inputs[i]);
        outputs[i] = (e2 - 1) / (e2 + 1);
    }
}

static void tanh_backward(const layer* l, const double* inputs, const double* deltas, double* outputs) {
    for (int i = 0; i < l->out_count; i++) {
        const double e2 = exp(2 * inputs[i]);
        const double t = (e2 - 1) / (e2 + 1);
        outputs[i] = 1 - t * t;
    }
}

static void relu_forward(const layer* l, const double* inputs, double* outputs) {
    for (int i = 0; i < l->out_count; i++) {
        outputs[i] = inputs[i] > 0 ? inputs[i] : 0;
    }
}

static void relu_backward(const layer* l, const double* inputs, const double* deltas, double* outputs) {
    for (int i = 0; i < l->out_count; i++) {
        outputs[i] = inputs[i] > 0 ? 1 : 0;
    }
}

static void leaky_relu_forward(const layer* l, const double* inputs, double* outputs) {
    for (int i = 0; i < l->out_count; i++) {
        outputs[i] = inputs[i] > 0 ? inputs[i] : LEAK * inputs[i];
    }
}

static void leaky_relu_backward(const layer* l, const double* inputs, const double* deltas, double* outputs) {
    for (int i = 0; i < l->out_count; i++) {
        outputs[i] = inputs[i] > 0 ? 1 : LEAK;
    }
}

static void silu_forward(const layer* l, const double* inputs, double* outputs) {
    for (int i = 0; i < l->out_count; i++) {
        const double sig = 1 / (1 + exp(-inputs[i]));
        outputs[i] = inputs[i] * sig * (1 - sig) + sig;
    }
}

static void silu_backward(const layer* l, const double* inputs, const double* deltas, double* outputs) {
    for (int i = 0; i < l->out_count; i++) {
        const double sig = 1 / (1 + exp(-inputs[i]));
        outputs[i] = inputs[i] * sig * (1 - sig) + sig;
    }
}

static void softmax_forward(const layer* l, const double* inputs, double* outputs) {
    double max = inputs[0];

    for (int i = 1; i < l->out_count; i++) {
        if (inputs[i] > max) max = inputs[i];
    }

    double sum = 0;

    for (int i = 0; i < l->out_count; i++) {
        const double e = exp(inputs[i] - max);
        sum += e;
        outputs[i] = e;
    }

    for (int i = 0; i < l->out_count; i++) {
        outputs[i] /= sum;
    }
}

//TODO
static void softmax_backward(const layer* l, const double* inputs, const double* deltas, double* outputs) {

}

layer_vtable store[] = {
    {sigmoid_forward, sigmoid_backward, activation_delta_to_gradients, apply_gradients_to_activation, default_layer_free},
    {tanh_forward, tanh_backward, activation_delta_to_gradients, apply_gradients_to_activation, default_layer_free},
    {relu_forward, relu_backward, activation_delta_to_gradients, apply_gradients_to_activation, default_layer_free},
    {leaky_relu_forward, leaky_relu_backward, activation_delta_to_gradients, apply_gradients_to_activation, default_layer_free},
    {silu_forward, silu_backward, activation_delta_to_gradients, apply_gradients_to_activation, default_layer_free},
    {softmax_forward, softmax_backward, activation_delta_to_gradients, apply_gradients_to_activation, default_layer_free}
};

inline layer* cnstr_activation_layer(const int type, const int outputCount) {
    layer* l = malloc(sizeof(layer));

    l->params = NULL;
    l->in_count = outputCount;
    l->out_count = outputCount;
    l->gradient_count = 0;

    l->vtable = store + type;
    l->initialize = no_initialization;

    return l;
}
