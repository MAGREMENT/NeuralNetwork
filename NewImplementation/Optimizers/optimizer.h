//
// Created by zacha on 21-10-25.
//

#ifndef NEWIMPLEMENTATION_OPTIMIZER_H
#define NEWIMPLEMENTATION_OPTIMIZER_H

typedef struct neural_network neural_network;

double** alloc_gradient_buffers(const neural_network* network, int initToZero);
void free_buffers(const neural_network* network, double** buffers);

typedef struct optimizer_args {
    double learning_rate;
    int layerIndex;
    int iteration;
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

void* cnstr_gradient_buffers_state(const optimizer* opt, const neural_network* network);
void free_gradient_buffers_state(void* state, const neural_network* network);

void* cnstr_double_gradient_buffers_state(const optimizer* opt, const neural_network* network);
void free_double_gradient_buffers_state(void* state, const neural_network* network);

void free_empty_opt(optimizer* opt);
void free_base_opt(optimizer* opt);

#endif //NEWIMPLEMENTATION_OPTIMIZER_H
