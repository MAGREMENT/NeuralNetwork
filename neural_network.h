#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

struct layer {
    int in_count;
    int out_count;
    double* weights;
    double* biases;
};

struct params {
    double learningRate;
    int activationType;
    int costType;
};

struct neural_network {
    int count;
    struct layer* layers;
    double learningRate;
    double (*activation)(double);
    double (*activationDerivative)(double);
    double (*cost)(double, double);
    double (*costDerivative)(double, double);
};

struct input_data {
    int count;
    double* values;
};

struct backpropagation_layer {
    int count;
    double* weightedInputs;
    double* afterActivations;
    double* nodeValues;
};

struct backpropagation_data {
    int count;
    struct backpropagation_layer* layers;
};

enum activation_type {
    DEFAULT = 0,
    SIGMOID = 1
};

enum cost_type {
    MEAN_SQUARED
};

struct neural_network* alloc_network(int count, const int numbers[]);
void apply_params(struct neural_network* network, struct params params);
void free_layers(struct layer* layers, int count);
void free_network(struct neural_network* network);
void randomize(struct neural_network* network, double from, double to);
void set_layer(struct layer layer, const double* weights, const double* biases);
struct input_data* alloc_input_data(int count);
void free_input_data(struct input_data* data);
void set_input_data(struct input_data data, const double values[]);
struct input_data* forward(struct layer layer, struct input_data input, double (*activation)(double));
void learn(struct neural_network* network, struct input_data data[], struct input_data expected[], int count);
struct backpropagation_data* alloc_back_data(const struct neural_network* network);
void free_back_data(struct backpropagation_data* data);
struct input_data* predict(struct neural_network* network, struct input_data* data);
void first_advance(struct layer layer, struct backpropagation_data* data, struct input_data* input, double (*activation)(double));
void continue_advance(struct layer layer, struct backpropagation_data* data, int inputIndex, double (*activation)(double));
struct backpropagation_data* traverse(const struct neural_network* network, struct input_data* data);
double cost(struct neural_network* network, struct input_data* data, struct input_data* expected);
double multi_cost(struct neural_network* network, struct input_data* data, struct input_data* expected, int count);
void apply_gradients(struct layer to, struct layer gradients, double learningRate);
void update_gradients(const struct neural_network* network, struct layer* gradients, struct input_data input, struct input_data expected);
struct layer* copy_layers(struct neural_network* network, int copyValues);
int is_valid(struct input_data* output, struct input_data* expected);

#endif // NEURAL_NETWORK_H
