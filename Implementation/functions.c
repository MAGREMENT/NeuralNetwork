#include <math.h>
#include <stdio.h>
#include "functions.h"

#include <float.h>
#include <stdlib.h>

#include "neural_network.h"
#include "utils.h"

inline double default_activation(double input, void* processedData){
    return input;
}

inline double derivative_default_activation(double input, void* processedData){
    return 1;
}

inline void* default_process_inputs(double* inputs, int count) {
    return NULL;
}

inline void default_free_data(void* data) {

}

inline double sigmoid_activation(double input, void* processedData) {
    return 1 / (1 + exp(-input));
}

inline double derivative_sigmoid_activation(double input, void* processedData){
    const double a = sigmoid_activation(input, processedData);
    return a * (1 - a);
}

inline double tanh_activation(double input, void* processedData) {
    const double e2 = exp(2 * input);
    return (e2 - 1) / (e2 + 1);
}

inline double derivative_tanh_activation(double input, void* processedData) {
    const double e2 = exp(2 * input);
    const double t = (e2 - 1) / (e2 + 1);
    return 1 - t * t;
}

inline double relu_activation(const double input, void* processedData) {
    return input > 0 ? input : 0;
}

inline double derivative_relu_activation(double input, void* processedData) {
    return input > 0 ? 1 : 0;
}

inline double silu_activation(double input, void* processedData) {
    return input / (1 + exp(-input));
}

inline double derivative_silu_activation(double input, void* processedData) {
    const double sig = 1 / (1 + exp(-input));
    return input * sig * (1 - sig) + sig;
}

inline double softmax_activation(double input, void* processedData) {
    return exp(input) / *(double*)processedData;
}

inline double derivative_softmax_activation(double input, void* processedData) {
    const double sum = *(double*)processedData;
    const double ex = exp(input);
    return (ex * sum - ex * ex) / (sum * sum);
}

void* softmax_process_inputs(double* inputs, int count) {
    double sum = 0;
    for(int i = 0; i < count; i++) {
        sum += exp(inputs[i]);
    }

    double* result = malloc(sizeof(double));
    *result = sum;
    return result;
}

void softmax_free_data(void* data) {
    free(data);
}

inline double mean_square_cost(double predicted, double expected){
    const double error = predicted - expected;
    return 0.5 * error * error;
}

inline double derivative_mean_square_cost(double predicted, double expected){
    return predicted - expected;
}

inline double cross_entropy_cost(double predicted, double expected) {
    const double v = expected >= 1 ? predicted : 1 - predicted;
    if (v <= 0) return 0;
    return -log(v);
}

inline double derivative_cross_entropy_cost(double predicted, double expected) {
    if (predicted == 0 || predicted == 1) return 0;
    return (expected - predicted) / (predicted * (predicted - 1));
}

inline void random_initialization(layer* layer) {
    const int total = layer->in_count * layer->out_count;
    for (int i = 0; i < total; i++) {
        layer->weights[i] = rand_std_nrml_distribution() * 0.01;
    }
}

inline void he_initialization(layer* layer) {
    const int total = layer->in_count * layer->out_count;
    const double scale = sqrt(2.0 / layer->in_count);
    for (int i = 0; i < total; i++) {
        layer->weights[i] = rand_std_nrml_distribution() * scale;
    }
}

inline void xavier_initialization(layer* layer) {
    const int total = layer->in_count * layer->out_count;
    const double scale = sqrt(1.0 / layer->in_count);
    for (int i = 0; i < total; i++) {
        layer->weights[i] = rand_std_nrml_distribution() * scale;
    }
}

inline void standardize(test_data* data) {
    //TODO
}

inline void min_max_scale(test_data* data) {
    if (data->count == 0) return;

    const int size = data->inputs[0].count;
    double* min = malloc(sizeof(double) * size);
    double* max = malloc(sizeof(double) * size);

    for (int i = 0; i < size; i++) {
        min[i] = DBL_MAX;
        max[i] = DBL_MIN;
    }

    for (int i = 0; i < data->count; i++) {
        input_data curr = data->inputs[i];
        if (curr.count != size) {
            free(min);
            free(max);
            return;
        }

        for (int j = 0; j < size; j++) {
            if (curr.values[j] < min[j]) min[j] = curr.values[j];
            if (curr.values[j] > max[j]) max[j] = curr.values[j];
        }
    }

    for (int i = 0; i < data->count; i++) {
        input_data curr = data->inputs[i];

        for (int j = 0; j < size; j++) {
            if (max[j] - min[j] == 0) curr.values[j] = 0;
            curr.values[j] = (curr.values[j] - min[j]) / (max[j] - min[j]);
        }
    }

    free(min);
    free(max);
}

//Cut Functions---------------------------------------------------------------------------------------------------------

inline int diagonal_cut(const double x, const double y) {
    return x > y ? 1 : 0;
}

inline int parable_10_cut(const double x, const double y) {
    return 0.05 * x * x + 8 > y ? 1 : 0;
}

inline int sinus_cut(const double x, const double y) {
    return sin(x) > y ? 1 : 0;
}


