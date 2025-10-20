//
// Created by zacha on 20-10-25.
//

#include "activation_layer.h"

#include <math.h>
#include <stdlib.h>

static double sigmoid(const double input) {
    return 1 / (1 + exp(-input));
}

static double* sigmoid_forward(const layer* l, double* inputs, int* didAllocate) {
    const int count = *(int*)l->params;
    for (int i = 0; i < count; i++) {
        inputs[i] = sigmoid(inputs[i]);
    }

    *didAllocate = false;
    return inputs;
}

static double* sigmoid_backward(const layer* l, double* inputs, double* deltas) {
    const int count = *(int*)l->params;
    for (int i = 0; i < count; i++) {
        const double a = sigmoid(inputs[i]);
        deltas[i] *= a * (1 - a);
    }

    return inputs;
}

layer_functions store[] = {
    {sigmoid_forward, sigmoid_backward, no_initialization, default_layer_free}
};

inline layer* cnstr_activation_layer(const int type, const int outputCount) {
    layer* l = malloc(sizeof(layer));
    int* oc = malloc(sizeof(int));
    *oc = outputCount;

    l->params = oc;
    l->functions = store[type];

    return l;
}
