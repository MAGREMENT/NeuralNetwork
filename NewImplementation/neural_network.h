//
// Created by zacha on 20-10-25.
//

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Iterators/iterator.h"
#include "Layers/layer.h"

typedef struct test_data {
    double* inputs;
    double* expected;
    int count;
} test_data ;

typedef struct neural_network {
    int layerCount;
    layer** layers;

    //TODO implement
    int threadCount;

    optimizer* optimizer;

    void (*get_cost_deltas)(const double* predicted, const double* expected, double* result, int count);
    double (*get_cost)(const double* predicted, const double* expected, int count);
} neural_network;

neural_network* alloc_neural_network(int layerCount, layer** layers);
void predict(const neural_network* network, const double* inputs, double* outputs);
void learn(const neural_network* network, test_data data, range range, optimizer_args args);
void initialize(const neural_network* network);

#endif //NEURAL_NETWORK_H
