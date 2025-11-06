//
// Created by zacha on 21-10-25.
//

#ifndef COST_H
#define COST_H

enum cost_types {
    MEAN_SQUARE,
    MEAN_ABSOLUTE,
    MEAN_LOG_COSH,
    BINARY_CROSS_ENTROPY
};

typedef struct cost_vtable {
    double (*get_cost)(const double* predicted, const double* expected, int count);
    void (*get_cost_deltas)(const double* predicted, const double* expected, double* result, int count);
} cost_vtable;

extern cost_vtable cost_vtables[];

extern cost_vtable softmax_bce_cost_vtable;

double mean_square_cost(const double* predicted, const double* expected, int count);

void derivative_mean_square_cost(const double* predicted, const double* expected, double* result, int count);

double mean_absolute_cost(const double* predicted, const double* expected, int count);

void derivative_mean_absolute_cost(const double* predicted, const double* expected, double* result, int count);

double mean_log_cosh_cost(const double* predicted, const double* expected, int count);

void derivative_mean_log_cosh_cost(const double* predicted, const double* expected, double* result, int count);

double binary_cross_entropy_cost(const double* predicted, const double* expected, int count);

void derivative_binary_cross_entropy_cost(const double* predicted, const double* expected, double* result, int count);

void derivative_softmax_bce_cost(const double* predicted, const double* expected, double* result, int count);

#endif //COST_H
