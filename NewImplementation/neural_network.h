//
// Created by zacha on 20-10-25.
//

#ifndef NEWIMPLEMENTATION_NEURAL_NETWORK_H
#define NEWIMPLEMENTATION_NEURAL_NETWORK_H

#include "cost.h"
#include "DataSelector/data_selector.h"
#include "Iterators/iterator.h"
#include "Optimizers/optimizer.h"
#include "Layers/layer.h"
#include "Schedulers/scheduler.h"

typedef struct learning_args {
    double learningRate;
    void* optimizer_state;
} learning_args;

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

    double learningRate;
    int shuffleDataOnIteration;

    optimizer* optimizer;
    scheduler* scheduler;
    data_selector* data_selector;

    cost_vtable* cost_vtable;
} neural_network;

extern neural_network* alloc_neural_network(int layerCount);
extern void free_neural_network(neural_network* network, int freeConstructed);

extern int get_in_count(const neural_network* network);
extern int get_out_count(const neural_network* network);

extern double** alloc_gradient_buffers(const neural_network* network, int initToZero);
extern void free_buffers(const neural_network* network, double** buffers);

extern void predict(const neural_network* network, const double* inputs, double* outputs);
extern void learn(const neural_network* network, test_data data, range range, learning_args args);
extern void iterative_learn(const neural_network* network, test_data data, learning_state* state, int iterations);

extern learning_state* alloc_state(const neural_network* network);
extern void free_state(const neural_network* network, learning_state* state);

extern void initialize(const neural_network* network);

extern double get_cost(const neural_network* network, const double* inputs, const double* expected);
extern double get_avg_cost(const neural_network* network, test_data data);

extern double get_binary_accuracy(const neural_network* network, test_data test);
extern double get_classification_accuracy(const neural_network* network, test_data test);

extern void shuffle_test_data(test_data test, const neural_network* network, int times);

#endif //NEWIMPLEMENTATION_NEURAL_NETWORK_H
