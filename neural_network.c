#include <stdlib.h>
#include "neural_network.h"

#include "functions.h"
#include "utils.h"

inline neural_network* alloc_network(const int count, const int numbers[]){
    neural_network* result = malloc(sizeof(neural_network));
    result->count = count - 1;
    result->layers = malloc(result->count * sizeof(layer));

    for(int i = 1; i < count; i++){
        const int in = numbers[i - 1];
        const int out = numbers[i];

        result->layers[i - 1].in_count = in;
        result->layers[i - 1].out_count = out;
        result->layers[i - 1].weights = malloc(in * out * sizeof(double));
        result->layers[i - 1].biases = malloc(out * sizeof(double));
    }

    return result;
}

inline void apply_params(neural_network* network, params params){
    network->learningRate = params.learningRate;
    for(int i = 0; i < network->count; i++) {
        switch (params.activationType) {
            case DEFAULT:
                network->layers[i].activation = default_activation;
            network->layers[i].activationDerivative = default_activation;
            break;
            case SIGMOID :
                network->layers[i].activation = sigmoid_activation;
            network->layers[i].activationDerivative = derivative_sigmoid_activation;
            break;
            default:
                network->layers[i].activation = NULL;
            network->layers[i].activationDerivative = NULL;
        }
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

inline void randomize(neural_network* network, double from, double to) {
    for(int n = 0; n < network->count; n++) {
        for(int o = 0; o < network->layers[n].out_count; o++) {
            for(int i = 0; i < network->layers[n].in_count; i++) {
                network->layers[n].weights[i * network->layers[n].out_count + o] = random(from, to);
            }

            network->layers[n].biases[o] = random(from, to);
        }
    }
}

inline void free_layers(layer* layers, int count) {
    for(int i = 0; i < count; i++){
        free(layers[i].biases);
        free(layers[i].weights);
    }

    free(layers);
}

inline void free_network(neural_network* network){
    free_layers(network->layers, network->count);
    free(network);
}

inline void set_layer(layer layer, const double* weights, const double* biases){
    for(int i = 0; i < layer.in_count; i++){
        for(int j = 0; j < layer.out_count; j++){
            layer.weights[i * layer.out_count + j] = weights[i * layer.out_count + j];
        }
    }

    for(int i = 0; i < layer.out_count; i++){
        layer.biases[i] = biases[i];
    }
}

inline input_data* alloc_input_data(int count){
    input_data* result = malloc(sizeof(input_data));

    result->count = count;
    result->values = malloc(count * sizeof(double));
    return result;
}

inline void free_input_data(input_data* data){
    free(data->values);
    free(data);
}

inline void set_input_data(input_data data, const double values[]){
    for(int i = 0; i < data.count; i++){
        data.values[i] = values[i];
    }
}

inline backpropagation_data* alloc_back_data(const neural_network* network) {
    backpropagation_data* result = malloc(network->count * sizeof(backpropagation_data));

    for(int i = 0; i < network->count; i++){
        const int n = network->layers[i].out_count;
        result[i].count = n;
        result[i].weightedInputs = malloc(n * sizeof(double));
        result[i].afterActivations = malloc(n * sizeof(double));
        result[i].nodeValues = malloc(n * sizeof(double));
    }

    return result;
}

inline void free_back_data(backpropagation_data* data, const int count){
    for(int i = 0; i < count; i++){
        free(data[i].weightedInputs);
        free(data[i].afterActivations);
        free(data[i].nodeValues);
    }

    free(data);
}

inline input_data* forward(layer layer, input_data input){
    input_data* result = alloc_input_data(layer.out_count);
    for(int i = 0; i < layer.out_count; i++){
        double n = layer.biases[i];
        for(int j = 0; j < layer.in_count; j++){
            n += input.values[j] * layer.weights[j * layer.out_count + i];
        }

        result->values[i] = layer.activation(n);
    }

    return result;
}

inline input_data* predict(neural_network* network, input_data* data){
    input_data* input = data;
    for(int i = 0; i < network->count; i++){
        input_data* before = input;
        input = forward(network->layers[i], *input);
        if(i != 0) free_input_data(before);
    }

    return input;
}

inline void continue_advance(layer layer, backpropagation_data* data, const int inputIndex){

    for(int i = 0; i < layer.out_count; i++){
        double n = layer.biases[i];
        for(int j = 0; j < layer.in_count; j++){
            n += data[inputIndex].afterActivations[j] * layer.weights[j * layer.out_count + i];
        }

        data[inputIndex + 1].weightedInputs[i] = n;
        data[inputIndex + 1].afterActivations[i] = layer.activation(n);
    }
}

inline void first_advance(layer layer, backpropagation_data* data, input_data* input){

    for(int i = 0; i < layer.out_count; i++){
        double n = layer.biases[i];
        for(int j = 0; j < layer.in_count; j++){
            n += input->values[j] * layer.weights[j * layer.out_count + i];
        }

        data[0].weightedInputs[i] = n;
        data[0].afterActivations[i] = layer.activation(n);
    }
}

inline backpropagation_data* traverse(const neural_network* network, input_data* data){
    backpropagation_data* result = alloc_back_data(network);

    first_advance(network->layers[0], result, data);
    for(int i = 0; i < network->count - 1; i++) {
        continue_advance(network->layers[i + 1], result, i);
    }

    return result;
}

inline double cost(neural_network* network, input_data* data, input_data* expected) {
    input_data* result = predict(network, data);
    double cost = 0;
    for(int i = 0; i < expected->count; i++) {
        cost += network->cost(result->values[i], expected->values[i]);
    }

    free_input_data(result);
    return cost;
}

inline double multi_cost(neural_network* network, input_data* data, input_data* expected, const int count) {
    double c = 0;
    for(int i = 0; i < count; i++) {
        c += cost(network, &data[i], &expected[i]);
    }

    return c;
}

inline void apply_gradients(layer to, gradients gradients, double learningRate){
    for(int i = 0; i < to.in_count; i++){
        for(int j = 0; j < to.out_count; j++){
            to.weights[i * to.out_count + j] -= gradients.weights[i * to.out_count + j] * learningRate;
        }
    }

    for(int i = 0; i < to.out_count; i++){
        to.biases[i] -= gradients.biases[i] * learningRate;
    }
}

inline void update_gradients(const neural_network* network, gradients* gradients, input_data input,
        const input_data expected) {

    backpropagation_data* data = traverse(network, &input);
    const int lastIndex = network->count - 1;

    for(int n = lastIndex; n >= 0; n--) {
        if(n == lastIndex) {
            for(int i = 0; i < expected.count; i++){
                const double costDerivative = network->costDerivative(data[n].afterActivations[i], expected.values[i]);
                const double activationDerivative = network->layers[n].activationDerivative(data[n].weightedInputs[i]);
                data[n].nodeValues[i] = activationDerivative * costDerivative;
            }
        }
        else {
            const int out = network->layers[n + 1].out_count;
            for(int i = 0; i < network->layers[n].out_count; i++) {
                double value = 0;
                for(int o = 0; o < out; o++) {
                    const double w = network->layers[n + 1].weights[i * out + o];
                    const double nv = data[n + 1].nodeValues[o];
                    value += nv * w;
                }

                data[n].nodeValues[i] = value * network->layers[n].activationDerivative(data[n].weightedInputs[i]);
            }
        }

        const layer current = network->layers[n];
        for(int o = 0; o < current.out_count; o++) {
            const double nv = data[n].nodeValues[o];
            for(int i = 0; i < current.in_count; i++) {
                const double g = nv * (n == 0 ? input.values[i] : data[n - 1].afterActivations[i]);
                gradients[n].weights[i * current.out_count + o] += g;
            }

            gradients[n].biases[o] += nv;
        }
    }

    free_back_data(data, network->count);
}

inline gradients* alloc_gradients(neural_network* network, int copyValues) {
    gradients* result = malloc(network->count * sizeof(gradients));

    for(int n = 0; n < network->count; n++) {
        const int in = network->layers[n].in_count;
        const int out = network->layers[n].out_count;

        result[n].biases = malloc(out * sizeof(double));
        result[n].weights = malloc(in * out * sizeof(double));

        for(int o = 0; o < out; o++) {
            for(int i = 0; i < in; i++) {
                result[n].weights[i * out + o] = copyValues ? network->layers[n].weights[i * out + o] : 0;
            }

            result[n].biases[o] = copyValues ? network->layers[n].biases[0] : 0;
        }
    }

    return result;
}

inline void free_gradients(gradients* gradients, const int count) {
    for(int i = 0; i < count; i++){
        free(gradients[i].biases);
        free(gradients[i].weights);
    }

    free(gradients);
}

inline void learn(neural_network* network, input_data data[], input_data expected[],
        const int from, const int to, const int then){
    gradients* gradients = alloc_gradients(network, 0);

    for(int i = from; i < to; i++){
        update_gradients(network, gradients, data[i], expected[i]);
    }

    for(int i = 0; i < then; i++) {
        update_gradients(network, gradients, data[i], expected[i]);
    }

    for(int i = 0; i < network->count; i++){
        apply_gradients(network->layers[i], gradients[i], network->learningRate / (to - from + then));
    }

    free_gradients(gradients, network->count);
}

inline int is_valid(input_data* output, input_data* expected) {
    return max_index(output->values, output->count) == max_index(expected->values, expected->count);
}
