#ifndef OLD_NN_H
#define OLD_NN_H

#include "data_selector.h"
#include "old_layer.h"
#include "learning_rate_sechduler.h"
#include "list.h"
#include "optimizer.h"
#include "Iterators/iterator.h"

typedef struct old_nn {
    int count;
    old_layer* layers;

    double learningRate;
    learning_rate_scheduler* scheduler;

    int shuffleDataOnIteration;
    int threadCount;
    data_selector* data_selector;
    optimizer* optimizer;

    double (*cost)(double, double);
    double (*costDerivative)(double, double);

    int outputLayerDeltaOptimization;
} old_nn;

typedef struct input_data {
    int count;
    double* values;
} input_data;

typedef struct backpropagation_data {
    int count;
    double* weightedInputs;
    double* afterActivations;
    double* nodeValues;
} backpropagation_data;

typedef struct test_data {
    int count;
    input_data* inputs;
    input_data* expected;
} test_data;

typedef struct test_result {
    double accuracy;
    double cost;
} test_result;

typedef struct learning_state {
    int iteration;
    void* optimizerState;
} learning_state;

typedef struct gradient_scale {
    int lower;
    int upper;
    int count;
} gradient_scale;

typedef struct weight_scale {
    int layer;
    int in;
    int out;
    int scale;
} weight_scale;

typedef struct bias_scale {
    int layer;
    int out;
    int scale;
} bias_scale;

typedef struct gradient_diagnostic {
    list* scales;
    list* criticalWeights;
    list* criticalBiases;
} gradient_diagnostic;

enum activation_type {
    DEFAULT,
    SIGMOID,
    TANH,
    RELU,
    LEAKY_RELU,
    SILU,
    SOFTMAX,
};

enum cost_type {
    MEAN_SQUARED,
    MEAN_ABSOLUTE,
    MEAN_LOG_COSH,
    BINARY_CROSS_ENTROPY
};

old_nn* alloc_network(int count, const int numbers[]);
void apply_default_hyper_params(old_nn* network);
void free_network(old_nn* network);

void set_activation_type(old_nn* network, int type, int outputType);
void set_cost_type(old_nn* network, int type);
void set_optimizer(old_nn* n, optimizer* opt);
void set_data_selector(old_nn* n, data_selector* ds);
void set_scheduler(old_nn* n, learning_rate_scheduler* lrs);

void get_activation_type(old_nn* network, int* type, int* outputType);
int get_cost_type(old_nn* network);

void old_initialize(old_nn* network);
void set_all_weights_and_biases(old_nn* network, double weights, double biases);

learning_state* alloc_state(old_nn* network);
void free_state(old_nn* network, learning_state* state);
void learn(old_nn* network, test_data* data, range range, double learningRate, void* optimizerState);
void iterative_learn(old_nn* network, test_data* data, learning_state* state, int iterations);

input_data* alloc_predict(old_nn* network, input_data* data);
void predict(old_nn* network, input_data* data, input_data* result);

/**
 * Traverse all layers to create backpropagation data
 * @param network
 * @param data
 * @return
 */
void traverse(const old_nn* network, input_data* data, backpropagation_data* result);
backpropagation_data* alloc_traverse(const old_nn* network, input_data* data);
double cost(old_nn* network, input_data* data, input_data* expected);
double avg_cost(old_nn* network, test_data* data);

void set_layer(old_layer layer, const double* weights, const double* biases);
void free_layers(old_layer* layers, int count);
void old_forward(old_layer layer, input_data input, input_data* result);
void first_advance(old_layer layer, const backpropagation_data* data, const input_data* input);
void continue_advance(old_layer layer, const backpropagation_data* data, int inputIndex);

void add_gradients(const old_nn* network, const layer_data* gradients, input_data input,
    input_data expected);
void async_add_gradients(const old_nn* network, const layer_data* gradients, input_data input,
    input_data expected, void* criticalSection);

input_data* alloc_input_data(int count);
input_data* alloc_input_data_array(int innerCount, int count);
void free_input_data(input_data* data);
void free_input_data_array(input_data* data, int count);
int is_valid(input_data* output, input_data* expected);

backpropagation_data* alloc_back_data(const old_nn* network);
void free_back_data(backpropagation_data* data, int count);

test_data* alloc_test_data(int count, int inputCount, int outputCount);
void free_test_data(test_data* data);
test_data* alloc_transfer_flattened_data(double* inputs, int inputCutoff, double* expected, int expectedCutoff, int count);
void free_transferred_flattened_data(test_data* data);

void shuffle_test_data(test_data *test, int count);
test_result test_network(old_nn* network, test_data *test);

gradient_diagnostic* alloc_run_gradient_diagnostic(old_nn* network, test_data* data, int vanishingBound,
    int explodingBound);
void free_gradient_diagnostic(gradient_diagnostic* diag);
void print_diagnostic(old_nn* network, gradient_diagnostic* diag);

#endif
