//
// Created by zacha on 01-10-25.
//

#include "optimizer.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define EPSILON 1e-8

static void apply_gradients(optimizer* opt, void* state, layer* layers, layer_data* gradients, const int layerCount,
                            int iteration, const double learningRate){
    for (int l = 0; l < layerCount; l++) {
        const int inCount = layers[l].in_count;
        const int outCount = layers[l].out_count;

        for(int o = 0; o < outCount; o++){
            for(int i = 0; i < inCount; i++){
                const int index = i * outCount + o;
                layers[l].weights[index] -= gradients[l].weights[index] * learningRate;
            }

            layers[l].biases[o] -= gradients[l].biases[o] * learningRate;
        }
    }
}

static void* create_empty_state(optimizer* opt, layer* layers, int layerCount) {
    return NULL;
}

static void do_nothing(void* state, int layerCount) {

}

inline optimizer* create_gradient_descent_optimizer() {
    optimizer* opt = malloc(sizeof(optimizer));

    opt->create_state = create_empty_state;
    opt->free_state = do_nothing;
    opt->apply_gradients = apply_gradients;
    opt->free = free;

    return opt;
}

static void apply_gradient_momentum(double* v, double* p, const double* g, const int index, const double momentum, const double lr) {
    const double velocity = v[index] * momentum + g[index];

    v[index] = velocity;
    p[index] -= lr * velocity;
}

static void apply_gradients_momentum(optimizer* opt, void* state, layer* layers, layer_data* gradients, int layerCount,
        int iteration, double learningRate) {
    const double momentum = *(double*) opt->params;
    const layer_data* v = state;

    for (int l = 0; l < layerCount; l++) {
        const int inCount = layers[l].in_count;
        const int outCount = layers[l].out_count;

        for(int o = 0; o < outCount; o++){
            for(int i = 0; i < inCount; i++){
                const int index = i * outCount + o;
                apply_gradient_momentum(v[l].weights, layers[l].weights, gradients[l].weights, index, momentum, learningRate);
            }

            apply_gradient_momentum(v[l].biases, layers[l].biases, gradients[l].biases, o, momentum, learningRate);
        }
    }
}

static void* create_momentum_state(optimizer* opt, layer* layers, int layerCount) {
    return alloc_layer_data_array(layers, layerCount, 0);
}

static void free_momentum_state(void* state, int layerCount) {
    free_layer_data_array(state, layerCount);
}

static void free_opt(optimizer* opt) {
    free(opt->params);
    free(opt);
}

inline optimizer* create_momentum_gradient_descent_optimizer(const double momentum) {
    optimizer* opt = malloc(sizeof(optimizer));
    double* p = malloc(sizeof(double));
    *p = momentum;

    opt->params = p;
    opt->create_state = create_momentum_state;
    opt->free_state = free_momentum_state;
    opt->apply_gradients = apply_gradients_momentum;
    opt->free = free_opt;

    return opt;
}

static void apply_gradient_nesterov(double* v, double* p, const double* g, const int index, const double momentum, const double lr) {
    const double velocity = v[index] * momentum + g[index];

    v[index] = velocity;
    p[index] -= lr * (velocity * momentum + g[index]);
}

static void apply_gradients_nesterov(optimizer* opt, void* state, layer* layers, layer_data* gradients, int layerCount,
        int iteration, double learningRate) {
    const double momentum = *(double*) opt->params;
    const layer_data* v = state;

    for (int l = 0; l < layerCount; l++) {
        const int inCount = layers[l].in_count;
        const int outCount = layers[l].out_count;

        for(int o = 0; o < outCount; o++){
            for(int i = 0; i < inCount; i++){
                const int index = i * outCount + o;
                apply_gradient_nesterov(v[l].weights, layers[l].weights, gradients[l].weights, index, momentum, learningRate);
            }

            apply_gradient_nesterov(v[l].biases, layers[l].biases, gradients[l].biases, o, momentum, learningRate);
        }
    }
}

inline optimizer* create_nesterov_optimizer(const double momentum) {
    optimizer* opt = malloc(sizeof(optimizer));
    double* p = malloc(sizeof(double));
    *p = momentum;

    opt->params = p;
    opt->create_state = create_momentum_state;
    opt->free_state = free_momentum_state;
    opt->apply_gradients = apply_gradients_nesterov;
    opt->free = free_opt;

    return opt;
}

static void apply_gradient_adam(double* v1, double* v2, double* p, const double* g, const int index,
    const double beta1, const double beta2, const int iteration, const double lr) {

    const double grad = g[index];
    v1[index] = beta1 * v1[index] + (1 - beta1) * grad;
    v2[index] = beta2 * v2[index] + (1 - beta2) * grad * grad;

    const double m = v1[index] / (1 - pow(beta1, iteration));
    const double v = v2[index] / (1 - pow(beta2, iteration)); //TODO avoid repeated pow calls

    p[index] -= lr * m / (sqrt(v) + EPSILON);
}

static void apply_gradients_adam(optimizer* opt, void* state, layer* layers, layer_data* gradients, int layerCount,
        int iteration, const double learningRate) {
    double* betas = opt->params;
    const layer_data* v1 = state;
    const layer_data* v2 = &v1[layerCount];

    for (int l = 0; l < layerCount; l++) {
        const int inCount = layers[l].in_count;
        const int outCount = layers[l].out_count;

        for(int o = 0; o < outCount; o++){
            for(int i = 0; i < inCount; i++){
                const int index = i * outCount + o;
                apply_gradient_adam(v1->weights, v2->weights, layers[l].weights, gradients[l].weights, index,
                    betas[0], betas[1], iteration, learningRate);
            }

            apply_gradient_adam(v1->biases, v2->biases, layers[l].biases, gradients[l].biases, o,
                              betas[0], betas[1], iteration, learningRate);
        }
    }
}

static void* create_adam_state(optimizer* opt, layer* layers, int layerCount) {
    layer* buffer = malloc(layerCount * 2 * sizeof(layer));
    memcpy(buffer, layers, layerCount * sizeof(layer));
    memcpy(buffer + layerCount, layers, layerCount * sizeof(layer));

    const auto result = alloc_layer_data_array(buffer, layerCount * 2, 0);
    free(buffer);
    return result;
}

static void free_adam_state(void* state, int layerCount) {
    free_layer_data_array(state, layerCount * 2);
}

optimizer* create_adam_optimizer(double beta1, double beta2) {
    optimizer* opt = malloc(sizeof(optimizer));
    double* p = malloc(sizeof(double) * 2);
    p[0] = beta1;
    p[1] = beta2;

    opt->params = p;
    opt->create_state = create_adam_state;
    opt->free_state = free_adam_state;
    opt->apply_gradients = apply_gradients_adam;
    opt->free = free_opt;

    return opt;
}
