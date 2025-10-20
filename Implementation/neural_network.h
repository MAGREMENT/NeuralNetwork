//
// Created by zacha on 20-10-25.
//

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Layers/layer.h"

typedef struct neural_network {
    int layerCount;
    layer** layers;
} neural_network;

neural_network* alloc_neural_network(int layerCount, layer** layers);
double* forward(neural_network* network, double* inputs);
void initialize(neural_network* network);

#endif //NEURAL_NETWORK_H
