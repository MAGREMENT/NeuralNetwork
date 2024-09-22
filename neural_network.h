#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

typedef struct layer {
    int in_count;
    int out_count;
    double* weights;
    double* biases;
    double (*activation)(double);
    double (*activationDerivative)(double);
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

enum activation_type {
    DEFAULT = 0,
    SIGMOID = 1
};

enum cost_type {
    MEAN_SQUARED
};

neural_network* alloc_network(int count, const int numbers[]);
void free_network(neural_network* network);

input_data* alloc_input_data(int count);
void free_input_data(input_data* data);

void apply_params(neural_network* network, params params);
void free_layers(layer* layers, int count);
void randomize(neural_network* network, double from, double to);
void set_layer(layer layer, const double* weights, const double* biases);
void set_input_data(input_data data, const double values[]);
input_data* forward(layer layer, input_data input);
void learn(neural_network* network, input_data data[], input_data expected[], int from,
    int to, int then);
backpropagation_data* alloc_back_data(const neural_network* network);
void free_back_data(backpropagation_data* data, int count);
input_data* predict(neural_network* network, input_data* data);
void first_advance(layer layer, backpropagation_data* data, input_data* input);
void continue_advance(layer layer, backpropagation_data* data, int inputIndex);
backpropagation_data* traverse(const neural_network* network, input_data* data);
double cost(neural_network* network, input_data* data, input_data* expected);
double multi_cost(neural_network* network, input_data* data, input_data* expected, int count);
void apply_gradients(layer to, gradients gradients, double learningRate);
void update_gradients(const neural_network* network, gradients* gradients, input_data input,
    input_data expected);
gradients* alloc_gradients(neural_network* network, int copyValues);
void free_gradients(gradients* gradients, int count);
int is_valid(input_data* output, input_data* expected);

#endif // NEURAL_NETWORK_H
