//
// Created by zacha on 20-10-25.
//

#include "activation_layer.h"
#include "../../Util/math_util.h"

#include <math.h>
#include <stdlib.h>

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

layer_functions store[] = {
    {sigmoid_forward, sigmoid_backward, activation_delta_to_gradients, apply_gradients_to_activation, no_initialization, default_layer_free}
};

inline layer* cnstr_activation_layer(const int type, const int outputCount) {
    layer* l = malloc(sizeof(layer));

    l->params = NULL;
    l->in_count = outputCount;
    l->out_count = outputCount;
    l->functions = store[type];

    return l;
}
