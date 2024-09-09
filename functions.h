#ifndef FUNCTIONS_H
#define FUNCTIONS_H

double default_activation(double n);
double sigmoid_activation(double n);
double derivative_sigmoid_activation(double n);
double mean_square_cost(double predicted, double expected);
double derivative_mean_square_cost(double predicted, double expected);

#endif // FUNCTIONS_H
