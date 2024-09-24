#ifndef FUNCTIONS_H
#define FUNCTIONS_H

double default_activation(double input, void* processedData);
void* default_process_inputs(double* inputs, int count);
void default_free_data(void* data);

double sigmoid_activation(double input, void* processedData);
double derivative_sigmoid_activation(double input, void* processedData);

double tanh_activation(double input, void* processedData);
double derivative_tanh_activation(double input, void* processedData);

double relu_activation(double input, void* processedData);
double derivative_relu_activation(double input, void* processedData);

double silu_activation(double input, void* processedData);
double derivative_silu_activation(double input, void* processedData);

double softmax_activation(double input, void* processedData);
double derivative_softmax_activation(double input, void* processedData);
void* softmax_process_inputs(double* inputs, int count);
void softmax_free_data(void* data);

double mean_square_cost(double predicted, double expected);
double derivative_mean_square_cost(double predicted, double expected);

int diagonal_cut(double x, double y);
int parabole_cut_10(double x, double y);

#endif // FUNCTIONS_H
