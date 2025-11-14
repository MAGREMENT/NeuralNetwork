//
// Created by zacha on 15-10-25.
//

#include "functions.h"
#include "store.h"
#ifndef STORE_C
#define STORE_C

int activation_store_max = SOFTMAX;
int cost_store_max = BINARY_CROSS_ENTROPY;

activation_data activation_store[] = {
    {default_activation, derivative_default_activation, default_process_inputs, default_free_data, random_initialization},
    {sigmoid_activation, derivative_sigmoid_activation, default_process_inputs, default_free_data, xavier_initialization},
    {tanh_activation, derivative_tanh_activation, default_process_inputs, default_free_data, xavier_initialization},
    {relu_activation, derivative_relu_activation, default_process_inputs, default_free_data, he_initialization},
    {leaky_relu_activation, derivative_leaky_relu_activation, default_process_inputs, default_free_data, he_initialization},
    {silu_activation, derivative_silu_activation, default_process_inputs, default_free_data, random_initialization},
    {softmax_activation, derivative_softmax_activation, softmax_process_inputs, softmax_free_data, random_initialization}
};

cost_data cost_store[] = {
    {mean_square_cost, derivative_mean_square_cost},
    {mean_absolute_cost, derivative_mean_absolute_cost},
    {mean_log_cosh_cost, derivative_mean_log_cosh_cost},
    {binary_cross_entropy_cost, derivative_binary_cross_entropy_cost}
};

inline int find_activation(double (*activation)(double, void*)) {
    for (int i = 0; i <= activation_store_max; i++) {
        if (activation_store[i].activation == activation) return i;
    }

    return -1;
}

inline int find_cost(double (*cost)(double, double)) {
    for (int i = 0; i <= cost_store_max; i++) {
        if (cost_store[i].cost == cost) return i;
    }

    return -1;
}

#endif //STORE_C
