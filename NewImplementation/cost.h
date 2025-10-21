//
// Created by zacha on 21-10-25.
//

#ifndef COST_H
#define COST_H

double mean_square_cost(const double* predicted, const double* expected, int count);

void derivative_mean_square_cost(const double* predicted, const double* expected, double* result, int count);

double mean_absolute_cost(const double* predicted, const double* expected, int count);

void derivative_mean_absolute_cost(const double* predicted, const double* expected, double* result, int count);

double mean_log_cosh_cost(const double* predicted, const double* expected, int count);

void derivative_mean_log_cosh_cost(const double* predicted, const double* expected, double* result, int count);

double binary_cross_entropy_cost(const double* predicted, const double* expected, int count);

void derivative_binary_cross_entropy_cost(const double* predicted, const double* expected, double* result, int count);

#endif //COST_H
