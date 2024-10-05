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
    double learningRate;
    int activationType;
    int costType;
} params ;

typedef struct neural_network {
    int count;
    layer* layers;
    double learningRate;
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

typedef struct gradients {
    double* weights;
    double* biases;
} gradients;

typedef struct test_data {
    int count;
    input_data* inputs;
    input_data* expected;
} test_data;

typedef struct test_result {
    double accuracy;
    double cost;
} test_result;

typedef struct batch {
    int from;
    int to;
    int then;
} batch;

enum activation_type {
    DEFAULT,
    SIGMOID,
    TANH,
    RELU,
    SILU,
    SOFTMAX
};

enum cost_type {
    MEAN_SQUARED
};

neural_network* alloc_network(int count, const int numbers[]);
void free_network(neural_network* network);
void apply_params(neural_network* network, params params);
void randomize(neural_network* network, double min, double max);
void learn(neural_network* network, test_data* data, batch batch);
void multi_learn(neural_network* network, test_data* data, int batchSize, int count,
    void (*on_iteration_end)(neural_network* network, test_data* data, int i));
input_data* predict(neural_network* network, input_data* data);
backpropagation_data* traverse(const neural_network* network, input_data* data);
double cost(neural_network* network, input_data* data, input_data* expected);
double multi_cost(neural_network* network, test_data* data);

void set_layer(layer layer, const double* weights, const double* biases);
void free_layers(layer* layers, int count);
input_data* forward(layer layer, input_data input);
void first_advance(layer layer, backpropagation_data* data, input_data* input);
void continue_advance(layer layer, backpropagation_data* data, int inputIndex);

gradients* alloc_gradients(neural_network* network, int copyValues);
void free_gradients(gradients* gradients, int count);
void apply_gradients(layer to, gradients gradients, double learningRate);
void update_gradients(const neural_network* network, gradients* gradients, input_data input,
    input_data expected);

input_data* alloc_input_data(int count);
void free_input_data(input_data* data);
void set_input_data(input_data data, const double values[]);
int is_valid(input_data* output, input_data* expected);

backpropagation_data* alloc_back_data(const neural_network* network);
void free_back_data(backpropagation_data* data, int count);

test_data* alloc_test_data(int count);
void free_test_data(test_data* data);
test_result test_network(neural_network* network, test_data *test);

batch create_batch(int current, int size, int max);
batch full_batch(int max);

#endif // NEURAL_NETWORK_H
