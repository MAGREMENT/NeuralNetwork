#ifndef FUNCTIONS_H
#define FUNCTIONS_H

double default_activation(double n);
double sigmoid_activation(double n);
double derivative_sigmoid_activation(double n);
double mean_square_cost(double predicted, double expected);
double derivative_mean_square_cost(double predicted, double expected);

int diagonal_cut(double x, double y);
int parabole_cut_10(double x, double y);

#endif // FUNCTIONS_H
