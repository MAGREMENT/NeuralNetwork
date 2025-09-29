#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

typedef struct layer {
    int in_count;
    int out_count;
    double* weights;
    double* biases;
    double (*activation)(double, void*);
    double (*activationDerivative)(double, void*);
    void* (*processInputs)(double*, int);
    void (*freeData)(void*);
} layer;

typedef struct params {
    double initialLearningRate;
    double learningRateDecay;
    double regularization;
    double momentum;
    int activationType;
    int outputActivationType;
    int costType;
} params ;

typedef struct neural_network {
    int count;
    layer* layers;

    double initialLearningRate;
    double learningRateDecay;
    double regularization;
    double momentum;

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

typedef struct layer_data {
    double* weights;
    double* biases;
} layer_data;

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
    int iterations;
    layer_data* velocities;
} learning_state;

enum activation_type {
    DEFAULT,
    SIGMOID,
    TANH,
    RELU,
    SILU,
    SOFTMAX
};

enum cost_type {
    MEAN_SQUARED,
    CROSS_ENTROPY
};

neural_network* alloc_network(int count, const int numbers[]);
void free_network(neural_network* network);

void set_activation_type(neural_network* network, int type, int outputType);
void set_cost_type(neural_network* network, int type);
void apply_params(neural_network* network, params params);

void randomize(neural_network* network, double min, double max);
void set_all_weights_and_biases(neural_network* network, double weights, double biases);

learning_state* alloc_state(neural_network* network);
void free_state(learning_state* state, int layerCount);
void learn(neural_network* network, test_data* data, int start, int batchSize, double learningRate, layer_data* velocities);
void linear_batch_learn(neural_network* network, test_data* data, learning_state* state, int batchSize, int iterations);
void random_batch_focus_learn(neural_network* network, test_data* data, learning_state* state, int batchSize, int batchFocus, int iterations);

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
double multi_cost(neural_network* network, test_data* data);

void set_layer(layer layer, const double* weights, const double* biases);
void free_layers(layer* layers, int count);
void forward(layer layer, input_data input, input_data* result);
void first_advance(layer layer, const backpropagation_data* data, const input_data* input);
void continue_advance(layer layer, const backpropagation_data* data, int inputIndex);

/**
 * Allocates an array of layer data corresponding to the neural network
 * @param network
 * @param copyValues true if the values are copies of neural network weights and biases
 * @return
 */
layer_data* alloc_layer_data_array(neural_network* network, int copyValues);
void free_layer_data_array(layer_data* gradients, int count);

void apply_gradients(layer to, layer_data gradients, double learningRate);
void apply_gradients_with_velocities(layer to, layer_data gradients, layer_data velocities, double learningRate,
    double momentum, double regularization);
void update_gradients(const neural_network* network, const layer_data* gradients, input_data input,
    input_data expected);

input_data* alloc_input_data(int count);
input_data* alloc_input_datas(int innerCount, int count);
void free_input_data(input_data* data);
void free_input_datas(input_data* data, int count);
void set_input_data(input_data data, const double values[]);
int is_valid(input_data* output, input_data* expected);

backpropagation_data* alloc_back_data(const neural_network* network);
void free_back_data(backpropagation_data* data, int count);

test_data* alloc_test_data(int count, int inputCount, int outputCount);
test_data* alloc_flattened_test_data(double* inputs, int inputCutoff, double* expected, int expectedCutoff, int count);
void free_test_data(test_data* data);
test_result test_network(neural_network* network, test_data *test);

#endif // NEURAL_NETWORK_H
