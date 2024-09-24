#include <math.h>
#include <stdio.h>
#include "functions.h"

#include <stdlib.h>

inline double default_activation(double input, void* processedData){
    return input;
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

inline double relu_activation(double input, void* processedData) {
    const double n = input;
    return n < 0 ? n : 0;
}

inline double derivative_relu_activation(double input, void* processedData) {
    return input > 0 ? 1 : 0;
}

inline double silu_activation(double input, void* processedData) {
    const double n = input;
    return n / (1 + exp(-n));
}

inline double derivative_silu_activation(double input, void* processedData) {
    const double n = input;
    const double sig = 1 / (1 + exp(-n));
    return n * sig * (1 - sig) + sig;
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
    return error * error;
}

inline double derivative_mean_square_cost(double predicted, double expected){
    return 2 * (predicted - expected);
}

inline int diagonal_cut(const double x, const double y) {
    return x > y ? 1 : 0;
}

inline int parabole_cut_10(const double x, const double y) {
    return 0.05 * x * x + 8 > y ? 1 : 0;
}


