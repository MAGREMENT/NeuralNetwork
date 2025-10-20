//
// Created by zacha on 20-10-25.
//

#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "../layer.h"

layer* cnstr_dense_layer(int inputCount, int outputCount, void (*initialize)(layer* l));
void initialize_dense_to_zero(layer* l);
void initialize_dense_to_one(layer* l);

#endif //DENSE_LAYER_H
