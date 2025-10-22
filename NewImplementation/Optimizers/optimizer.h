//
// Created by zacha on 21-10-25.
//

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

typedef struct optimizer_args {
    double learning_rate;
} optimizer_args;

typedef struct optimizer optimizer;

struct optimizer {
    void* params;

    void (*apply_gradients)(double* to, const double* delta, optimizer_args args);
};

#endif //OPTIMIZER_H
