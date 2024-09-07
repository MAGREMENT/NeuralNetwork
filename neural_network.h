#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#include <malloc.h>
#include "functions.h"

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

inline struct neural_network* alloc_network(int count, const int numbers[]){
    struct neural_network* result = malloc(sizeof(struct neural_network));
    result->count = count - 1;
    result->layers = malloc(result->count * sizeof(struct layer));

    for(int i = 1; i < count; i++){
        int in = numbers[i - 1];
        int out = numbers[i];

        result->layers[i - 1].in_count = in;
        result->layers[i - 1].out_count = out;
        result->layers[i - 1].weights = malloc(in * out * sizeof(double));
        result->layers[i - 1].biases = malloc(out * sizeof(double));
    }

    return result;
}

inline void apply_params(struct neural_network* network, struct params params){
    network->learningRate = params.learningRate;
    switch (params.activationType) {
        case DEFAULT:
            network->activation = default_activation;
            network->activationDerivative = default_activation;
            break;
        case SIGMOID :
            network->activation = sigmoid_activation;
            network->activationDerivative = derivative_sigmoid_activation;
            break;
        default:
            network->activation = NULL;
            network->activationDerivative = NULL;
    }

    switch (params.costType) {
        case MEAN_SQUARED:
            network->cost = mean_square_cost;
            network->costDerivative = derivative_mean_square_cost;
            break;
        default:
            network->cost = NULL;
            network->costDerivative = NULL;
    }
}

inline void free_network(struct neural_network* network){
    for(int i = 0; i < network->count; i++){
        free(network->layers[i].biases);
        free(network->layers[i].weights);
    }

    free(network->layers);
    free(network);
}

inline void set_layer(struct layer layer, const double* weights, const double* biases){
    for(int i = 0; i < layer.in_count; i++){
        for(int j = 0; j < layer.out_count; j++){
            layer.weights[i * layer.out_count + j] = weights[i * layer.out_count + j];
        }
    }

    for(int i = 0; i < layer.out_count; i++){
        layer.biases[i] = biases[i];
    }
}

inline struct input_data* alloc_input_data(int count){
    struct input_data* result = malloc(sizeof(struct input_data));

    result->count = count;
    result->values = malloc(count * sizeof(double));
    return result;
}

inline void free_input_data(struct input_data* data){
    free(data->values);
    free(data);
}

inline void set_input_data(struct input_data data, const double values[]){
    for(int i = 0; i < data.count; i++){
        data.values[i] = values[i];
    }
}

inline struct backpropagation_data* alloc_back_data(struct neural_network* network){
    struct backpropagation_data* result = malloc(sizeof(struct backpropagation_data));

    result->count = network->count;
    result->layers = malloc(network->count * sizeof(struct backpropagation_layer));
    for(int i = 0; i < network->count; i++){
        int n = network->layers[i].out_count;
        result->layers[i].count = n;
        result->layers[i].weightedInputs = malloc(n * sizeof(double));
        result->layers[i].afterActivations = malloc(n * sizeof(double));
    }

    return result;
}

inline void free_back_data(struct backpropagation_data* data){
    for(int i = 0; i < data->count; i++){
        free(data->layers[i].weightedInputs);
        free(data->layers[i].afterActivations);
    }

    free(data->layers);
    free(data);
}

inline struct input_data* forward(struct layer layer, struct input_data input, double (*activation)(double)){
    struct input_data* result = alloc_input_data(layer.out_count);
    for(int i = 0; i < layer.out_count; i++){
        double n = layer.biases[i];
        for(int j = 0; j < layer.in_count; j++){
            n += input.values[j] * layer.weights[j * layer.out_count + i];
        }

        result->values[i] = activation(n);
    }

    return result;
}

inline struct backpropagation_data* traverse(struct neural_network* network, struct input_data* data){
    struct backpropagation_data* result = alloc_back_data(network);
    //TODO
    return result;
}

inline struct input_data* predict(struct neural_network* network, struct input_data* data){
    struct input_data* input = data;
    for(int i = 0; i < network->count; i++){
        struct input_data* before = input;
        input = forward(network->layers[i], *input, network->activation);
        if(i != 0) free_input_data(before);
    }

    return input;
}

inline void apply_gradients(struct layer to, struct layer gradients, double learningRate){
    if(to.in_count != gradients.in_count || to.out_count != gradients.out_count) return;

    for(int i = 0; i < to.in_count; i++){
        for(int j = 0; j < to.out_count; j++){
            to.weights[i * to.out_count + j] -= gradients.weights[i * to.out_count + j] * learningRate;
        }
    }

    for(int i = 0; i < to.out_count; i++){
        to.biases[i] -= gradients.biases[i] * learningRate;
    }
}

inline void update_gradients(struct neural_network* network, struct layer gradients[], struct input_data* data,
        struct input_data* expected) {
    struct backpropagation_data* backData = traverse(network, data);
    double nv[data->count];
    struct backpropagation_layer l = backData->layers[network->count - 1];
    for(int i = 0; i < data->count; i++){
        double costDerivative = network->costDerivative(l.afterActivations[i],expected->values[i]);
        double activationDerivative = network->activationDerivative(l.weightedInputs[i]);
        nv[i] = activationDerivative * costDerivative;
    }

    free_back_data(backData);
}

inline void learn(struct neural_network* network, struct input_data data[], int count){
    struct layer gradients[network->count];

    /*for(int i = 0; i < count; i++){
        update_gradients(network, gradients, data[i]);
    }

    for(int i = 0; i < network->count; i++){
        apply_gradients(network->layers[i], gradients[i], network->learningRate);
    }*/
}
#endif // NEURAL_NETWORK_H
