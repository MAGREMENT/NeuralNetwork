#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include "neural_network.h"

//Activation

double default_activation(double input, void* processedData);
double derivative_default_activation(double input, void* processedData);
void* default_process_inputs(double* inputs, int count);
void default_free_data(void* data);

double sigmoid_activation(double input, void* processedData);
double derivative_sigmoid_activation(double input, void* processedData);

double tanh_activation(double input, void* processedData);
double derivative_tanh_activation(double input, void* processedData);

double relu_activation(double input, void* processedData);
double derivative_relu_activation(double input, void* processedData);

double leaky_relu_activation(double input, void* processedData);
double derivative_leaky_relu_activation(double input, void* processedData);

double silu_activation(double input, void* processedData);
double derivative_silu_activation(double input, void* processedData);

double softmax_activation(double input, void* processedData);
double derivative_softmax_activation(double input, void* processedData);
void* softmax_process_inputs(double* inputs, int count);
void softmax_free_data(void* data);

//Cost (Loss)

double mean_square_cost(double predicted, double expected);
double derivative_mean_square_cost(double predicted, double expected);

double cross_entropy_cost(double predicted, double expected);
double derivative_cross_entropy_cost(double predicted, double expected);

//Initialization

void random_initialization(layer* layer);
void he_initialization(layer* layer);
void xavier_initialization(layer* layer);

//Normalization

void standardize(test_data* data);
void min_max_scale(test_data* data);

int diagonal_cut(double x, double y);
int parable_10_cut(double x, double y);
int sinus_cut(double x, double y);

#endif // FUNCTIONS_H
