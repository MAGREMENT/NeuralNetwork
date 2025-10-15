//
// Created by zacha on 15-10-25.
//

#ifndef STORE_H
#define STORE_H

#include "layer.h"
#include "neural_network.h"

typedef struct activation_data {
    double (*activation)(double, void*);
    double (*activationDerivative)(double, void*);
    void* (*processInputs)(double*, int);
    void (*freeData)(void*);
    void (*initialization)(layer* layer);
} activation_data;

typedef struct cost_data {
    double (*cost)(double, double);
    double (*costDerivative)(double, double);
} cost_data;

extern int activation_store_max;
extern int cost_store_max;

extern activation_data activation_store[];

extern cost_data cost_store[];

int find_activation(double (*activation)(double, void*));
int find_cost(double (*cost)(double, double));

#endif //STORE_H
