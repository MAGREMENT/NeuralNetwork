#include <math.h>
#include "functions.h"

inline double default_activation(const double n){
    return n;
}

inline double sigmoid_activation(double n) {
    return 1 / (1 + exp(-n));
}

inline double derivative_sigmoid_activation(double n){
    const double a = sigmoid_activation(n);
    return a * (1 - a);
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


