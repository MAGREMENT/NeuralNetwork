#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "data_selector.h"
#include "layer.h"
#include "learning_rate_sechduler.h"
#include "optimizer.h"
#include "Iterators/iterator.h"

typedef struct neural_network {
    int count;
    layer* layers;
    void (*initialization)(layer* layer);

    double learningRate;
    learning_rate_scheduler* scheduler;

    int shuffleDataOnIteration;
    data_selector* data_selector;
    optimizer* optimizer;

    double (*cost)(double, double);
    double (*costDerivative)(double, double);
} neural_network;

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
    gradient_scale* scales;
    int scaleCount;
    weight_scale* criticalWeights;
    int cwCount;
    bias_scale* criticalBiases;
    int cbCount;
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
    CROSS_ENTROPY
};

neural_network* alloc_network(int count, const int numbers[]);
void apply_default_hyper_params(neural_network* network);
void free_network(neural_network* network);

void set_activation_type(neural_network* network, int type, int outputType);
void set_cost_type(neural_network* network, int type);
void set_optimizer(neural_network* n, optimizer* opt);
void set_data_selector(neural_network* n, data_selector* ds);
void set_scheduler(neural_network* n, learning_rate_scheduler* lrs);

void initialize(neural_network* network);
void set_all_weights_and_biases(neural_network* network, double weights, double biases);

learning_state* alloc_state(neural_network* network);
void free_state(neural_network* network, learning_state* state);
void learn(neural_network* network, test_data* data, range range, double learningRate, void* optimizerState);
void iterative_learn(neural_network* network, test_data* data, learning_state* state, int iterations);

input_data* alloc_predict(neural_network* network, input_data* data);
void predict(neural_network* network, input_data* data, input_data* result);

/**
 * Traverse all layers to create backpropagation data
 * @param network
 * @param data
 * @return
 */
void traverse(const neural_network* network, input_data* data, backpropagation_data* result);
backpropagation_data* alloc_traverse(const neural_network* network, input_data* data);
double cost(neural_network* network, input_data* data, input_data* expected);
double avg_cost(neural_network* network, test_data* data);

void set_layer(layer layer, const double* weights, const double* biases);
void free_layers(layer* layers, int count);
void forward(layer layer, input_data input, input_data* result);
void first_advance(layer layer, const backpropagation_data* data, const input_data* input);
void continue_advance(layer layer, const backpropagation_data* data, int inputIndex);

void add_gradients(const neural_network* network, const layer_data* gradients, input_data input,
    input_data expected);

input_data* alloc_input_data(int count);
input_data* alloc_input_data_array(int innerCount, int count);
void free_input_data(input_data* data);
void free_input_data_array(input_data* data, int count);
int is_valid(input_data* output, input_data* expected);

backpropagation_data* alloc_back_data(const neural_network* network);
void free_back_data(backpropagation_data* data, int count);

test_data* alloc_test_data(int count, int inputCount, int outputCount);
test_data* alloc_flattened_test_data(double* inputs, int inputCutoff, double* expected, int expectedCutoff, int count);
void free_test_data(test_data* data);

void shuffle_test_data(test_data *test, int count);
test_result test_network(neural_network* network, test_data *test);

gradient_diagnostic* alloc_run_gradient_diagnostic(neural_network* network, test_data* data, int vanishingBound,
    int explodingBound);
void free_gradient_diagnostic(gradient_diagnostic* diag);
void print_diagnostic(neural_network* network, gradient_diagnostic* diag);

#endif // NEURAL_NETWORK_H
