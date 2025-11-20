//
// Created by zacha on 20-10-25.
//

#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "../layer.h"
#include "../../multi-threading.h"

layer* cnstr_dense_layer(int inputCount, int outputCount, void (*initialize)(const layer* l));
layer* cnstr_multi_thread_dense_layer(int inputCount, int outputCount, int thread_count, void (*initialize)(const layer* l));
layer* cnstr_worker_multi_thread_dense_layer(int inputCount, int outputCount, worker_context* context, void (*initialize)(const layer* l));

extern void set_weights(const layer* l, double values[]);
extern void set_biases(const layer* l, double values[]);
extern void set_all_weights_and_biases(const layer* l, double weights, double biases);

extern void initialize_dense_to_zero(const layer* l);
extern void initialize_dense_to_one(const layer* l);
extern void initialize_dense_to_two(const layer* l);
extern void initialize_dense_random(const layer* layer);
extern void initialize_dense_he(const layer* layer);
extern void initialize_dense_xavier(const layer* layer);

#endif //DENSE_LAYER_H
