//
// Created by zacha on 20-10-25.
//

#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "../layer.h"

layer* cnstr_dense_layer(int inputCount, int outputCount, void (*initialize)(const layer* l));

void initialize_dense_to_zero(const layer* l);
void initialize_dense_to_one(const layer* l);
void initialize_dense_random(const layer* layer);
void initialize_dense_he(const layer* layer);
void initialize_dense_xavier(const layer* layer);

#endif //DENSE_LAYER_H
