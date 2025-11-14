#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

#include "../layer.h"

enum activation_types {
    SIGMOID,
    TANH,
    RELU,
    LEAKY_RELU,
    SILU,
    SOFTMAX
};

extern layer* cnstr_activation_layer(int type, int outputCount);
extern layer* cnstr_softmax_bce_layer(int outputCount);

#endif //ACTIVATION_LAYER_H