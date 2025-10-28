//
// Created by zacha on 20-10-25.
//

#ifndef NEWIMPLEMENTATION_NEURAL_NETWORK_H
#define NEWIMPLEMENTATION_NEURAL_NETWORK_H

#include "cost.h"
#include "DataSelector/data_selector.h"
#include "Iterators/iterator.h"
#include "Layers/layer.h"
#include "Schedulers/scheduler.h"

typedef struct test_data {
    double* inputs;
    double* expected;
    int count;
} test_data;

typedef struct learning_state {
    int iteration;
    void* optimizerState;
} learning_state;

typedef struct neural_network {
    int layerCount;
    layer** layers;

    //TODO implement
    int threadCount;
    double learningRate;
    bool shuffleDataOnIteration;

    optimizer* optimizer;
    scheduler* scheduler;
    data_selector* data_selector;

    cost_vtable* cost_vtable;
} neural_network;

neural_network* alloc_neural_network(int layerCount);
void free_neural_network(neural_network* network, int freeConstructed);

int get_in_count(const neural_network* network);
int get_out_count(const neural_network* network);

void predict(const neural_network* network, const double* inputs, double* outputs);
void learn(const neural_network* network, test_data data, range range, optimizer_args args);
void iterative_learn(neural_network* network, test_data data, learning_state* state, int iterations);

learning_state* alloc_state(const neural_network* network);
void free_state(const neural_network* network, learning_state* state);

void initialize(const neural_network* network);

double get_cost(const neural_network* network, const double* inputs, const double* expected);
double get_avg_cost(const neural_network* network, test_data data);

#endif //NEWIMPLEMENTATION_NEURAL_NETWORK_H
