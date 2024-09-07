#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#include <math.h>

inline double default_activation(const double n){
    return n;
}

inline double sigmoid_activation(double n){
    return 1 / (1 + exp(-n));
}

inline double derivative_sigmoid_activation(double n){
    double a = sigmoid_activation(n);
    return a * (1 - a);
}

inline double mean_square_cost(double predicted, double expected){
    double error = predicted - expected;
    return error * error;
}

inline double derivative_mean_square_cost(double predicted, double expected){
    return 2 * (predicted - expected);
}
#endif // FUNCTIONS_H
