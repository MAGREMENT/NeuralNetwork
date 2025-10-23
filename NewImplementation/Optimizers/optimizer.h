//
// Created by zacha on 21-10-25.
//

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

typedef struct optimizer_args {
    double learning_rate;
} optimizer_args;

typedef struct optimizer optimizer;

typedef struct optimizer_vtable {
    void (*apply_gradients)(optimizer* opt, double* to, const double* gradients, int count, optimizer_args args);
    void (*free) (optimizer* opt);
} optimizer_vtable;

struct optimizer {
    void* params;

    optimizer_vtable* vtable;
};

#endif //OPTIMIZER_H
