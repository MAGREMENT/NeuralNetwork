//
// Created by zacha on 20-10-25.
//

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Layers/layer.h"

typedef struct neural_network {
    int layerCount;
    layer** layers;

    void (*get_cost_deltas)(const double* inputs, const double* expected, double* result, int count);
    double (*get_cost)(const double* inputs, const double* expected, int count);
} neural_network;

neural_network* alloc_neural_network(int layerCount, layer** layers);
double* predict(neural_network* network, double* inputs);
void learn(neural_network* network, const double* input, const double* expected, int count);
void initialize(neural_network* network);

#endif //NEURAL_NETWORK_H
