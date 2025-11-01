//
// Created by zacha on 21-10-25.
//

#include "cost.h"

#include <math.h>

#define COST_CLAMP 1e-12

inline double mean_square_cost(const double* predicted, const double* expected, int count){
    double result = 0;
    for (int i = 0; i < count; i++) {
        const double error = predicted[i] - expected[i];
        result += error * error;
    }
    
    return result;
}

inline void derivative_mean_square_cost(const double* predicted, const double* expected, double* result, int count){
    for (int i = 0; i < count; i++) {
        result[i] = 2 * (predicted[i] - expected[i]); 
    }
}

inline double mean_absolute_cost(const double* predicted, const double* expected, int count) {
    double result = 0;
    for (int i = 0; i < count; i++) {
        result += fabs(predicted[i] - expected[i]);
    }
    
    return result;
}

inline void derivative_mean_absolute_cost(const double* predicted, const double* expected, double* result, int count) {
    for (int i = 0; i < count; i++) {
        const double mean = predicted[i] - expected[i];
        if (mean > 0) result[i] = 1;
        else if (mean < 0) result[i] = -1;
        else result[i] = 0;
    }
}

inline double mean_log_cosh_cost(const double* predicted, const double* expected, int count) {
    double result = 0;
    for (int i = 0; i < count; i++) {
        result += log(cosh(predicted[i] - expected[i]));
    }
    
    return result;
}

inline void derivative_mean_log_cosh_cost(const double* predicted, const double* expected, double* result, int count) {
    for (int i = 0; i < count; i++) {
        result[i] = tanh(predicted[i] - expected[i]);
    }
}

inline double binary_cross_entropy_cost(const double* predicted, const double* expected, int count) {
    double result = 0;
    for (int i = 0; i < count; i++) {
        double v = expected[i] >= 1 ? predicted[i] : 1 - predicted[i];
        if (v <= 0) v = COST_CLAMP;
        result -= log(v);
    }
    
    return result;
}

inline void derivative_binary_cross_entropy_cost(const double* predicted, const double* expected, double* result, int count) {
    for (int i = 0; i < count; i++) {
        double p = predicted[i];
        if (p == 0) p = COST_CLAMP;
        else if (p == 1) p = 1 - COST_CLAMP;

        result[i] = (expected[i] - p) / (p * (p - 1));
    }
}

cost_vtable cost_vtables[] = {
    {mean_square_cost, derivative_mean_square_cost},
    {mean_absolute_cost, derivative_mean_absolute_cost},
    {mean_log_cosh_cost, derivative_mean_log_cosh_cost},
    {binary_cross_entropy_cost, derivative_binary_cross_entropy_cost}
};