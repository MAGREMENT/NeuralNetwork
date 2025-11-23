//
// Created by zacha on 20-10-25.
//

#ifndef NEWIMPLEMENTATION_NEURAL_NETWORK_H
#define NEWIMPLEMENTATION_NEURAL_NETWORK_H

#include "cost.h"
#include "multi-threading.h"
#include "DataSelector/data_selector.h"
#include "Iterators/iterator.h"
#include "Optimizers/optimizer.h"
#include "Layers/layer.h"
#include "Schedulers/scheduler.h"

typedef struct test_data {
    double* inputs;
    double* expected;
    int count;
} test_data;

typedef struct learning_buffers {
    double** gradient_buffers;
    double** iv_buffers;
} learning_buffers;

typedef struct learning_state {
    int iteration;
    void* optimizerState;
} learning_state;

typedef struct learning_data {
    learning_state state;
    learning_buffers* buffers;
} learning_data;

typedef struct neural_network_vtable {
    void (*free_params) (neural_network*);
} neural_network_vtable;

typedef struct neural_network {
    int layerCount;
    layer** layers;

    double learningRate;
    int shuffleDataOnIteration;

    optimizer* optimizer;
    scheduler* scheduler;
    data_selector* data_selector;

    cost_vtable* cost_vtable;

    thread_pool* thread_pool;
    parallel_range_executor* batch_executor;
} neural_network;

extern neural_network* alloc_neural_network(int layerCount);
extern void free_neural_network(neural_network* network, int freeConstructed);

extern int get_in_count(const neural_network* network);
extern int get_out_count(const neural_network* network);

extern double** alloc_gradient_buffers(const neural_network* network, int initToZero);
extern void free_buffers(const neural_network* network, double** buffers);

extern void predict(const neural_network* network, const double* inputs, double* outputs);
extern void learn(const neural_network* network, test_data data, iteration_range range, learning_data* ld, double learningRate);
extern void learn_stateless(const neural_network* network, test_data data, iteration_range range, double learningRate);
extern void iterative_learn(const neural_network* network, test_data data, learning_data* ld, int iterations);
extern void iterative_learn_stateless(const neural_network* network, test_data data, int iterations);

extern learning_data* alloc_learning_data(const neural_network* network);
extern void free_learning_data(const neural_network* network, learning_data* state);

extern void initialize(const neural_network* network);

extern double get_cost(const neural_network* network, const double* inputs, const double* expected);
extern double get_avg_cost(const neural_network* network, test_data data);

extern double get_binary_accuracy(const neural_network* network, test_data test);
extern double get_classification_accuracy(const neural_network* network, test_data test);

extern void shuffle_test_data(test_data test, const neural_network* network, int times);
extern void separate_test_data(test_data original, int inCutoff, int outCutoff, test_data* training, test_data* testing, double split);

extern int save_parameters(const neural_network* network, const char* file);
extern int restore_parameters(const neural_network* network, const char* file);

#endif //NEWIMPLEMENTATION_NEURAL_NETWORK_H
