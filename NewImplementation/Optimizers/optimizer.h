//
// Created by zacha on 21-10-25.
//

#ifndef NEWIMPLEMENTATION_OPTIMIZER_H
#define NEWIMPLEMENTATION_OPTIMIZER_H

typedef struct neural_network neural_network;

typedef struct optimizer_args {
    double learning_rate;
    void* state;
} optimizer_args;

typedef struct optimizer optimizer;

typedef struct optimizer_vtable {
    void* (*cnstr_state)(const optimizer* opt, const neural_network* network);
    void (*free_state)(void* state, const neural_network* network);
    void (*apply_gradients)(const optimizer* opt, double* to, const double* gradients, int count, optimizer_args args);
    void (*free) (optimizer* opt);
} optimizer_vtable;

struct optimizer {
    void* params;

    optimizer_vtable* vtable;
};

void* cnstr_empty_state(const optimizer* opt, const neural_network* network);
void free_empty_state(void* state, const neural_network* network);

#endif //NEWIMPLEMENTATION_OPTIMIZER_H
