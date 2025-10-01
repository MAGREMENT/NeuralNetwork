//
// Created by zacha on 01-10-25.
//

#include "optimizer.h"

#include <math.h>
#include <stdlib.h>

static void apply_gradients(optimizer* opt, void* state, layer to, layer_data gradients, double learningRate){
    for(int i = 0; i < to.in_count; i++){
        for(int j = 0; j < to.out_count; j++){
            const int index = i * to.out_count + j;
            to.weights[index] -= gradients.weights[index] * learningRate;
        }
    }

    for(int i = 0; i < to.out_count; i++){
        to.biases[i] -= gradients.biases[i] * learningRate;
    }
}

static void* create_empty_state(optimizer* opt) {
    return NULL;
}

static void do_nothing(void* state) {

}

optimizer* create_gradient_descent_optimizer() {
    optimizer* opt = malloc(sizeof(optimizer));

    opt->create_state = create_empty_state;
    opt->free_state = do_nothing;
    opt->apply_gradients = apply_gradients;
    opt->free = free;

    return opt;
}

static void apply_gradients_adam(layer to, layer_data gradients, layer_data avg_gradients,
    layer_data avg_sqr_gradients, int iteration, double learningRate, double beta1, double beta2){

    for(int i = 0; i < to.in_count; i++){
        for(int j = 0; j < to.out_count; j++){
            const int index = i * to.out_count + j;

            const double g = gradients.weights[index];
            avg_gradients.weights[index] = beta1 * avg_gradients.weights[index] + (1 - beta1) * g;
            avg_sqr_gradients.weights[index] = beta2 * avg_gradients.weights[index] + (1 - beta2) * g * g;

            const double m = avg_gradients.weights[index] / (1 - pow(beta1, iteration));
            const double v = avg_sqr_gradients.weights[index] / (1 - pow(beta2, iteration));

            to.weights[index] -= learningRate * m / (sqrt(v) + 0.00000001);
        }
    }

    for(int o = 0; o < to.out_count; o++){

        const double g = gradients.weights[o];
        avg_gradients.biases[o] = beta1 * avg_gradients.biases[o] + (1 - beta1) * g;
        avg_sqr_gradients.biases[o] = beta2 * avg_gradients.biases[o] + (1 - beta2) * g * g;

        const double m = avg_gradients.biases[o] / (1 - pow(beta1, iteration));
        const double v = avg_sqr_gradients.biases[o] / (1 - pow(beta2, iteration));

        to.biases[o] -= learningRate * m / (sqrt(v) + 0.00000001);
    }
}

static void apply_gradients_with_velocities(layer to, layer_data gradients, layer_data velocities, double learningRate,
                                            const double momentum, const double regularization){
    const double weightDecay = 1 - regularization * learningRate;

    for(int i = 0; i < to.in_count; i++){
        for(int j = 0; j < to.out_count; j++){
            const int index = i * to.out_count + j;
            const double velocity = velocities.weights[index] * momentum - gradients.weights[index] * learningRate;

            velocities.weights[index] = velocity;
            to.weights[index] = to.weights[index] * weightDecay + velocity;
        }
    }

    for(int i = 0; i < to.out_count; i++){
        const double velocity = velocities.biases[i] * momentum - gradients.biases[i] * learningRate;

        velocities.biases[i] = velocity;
        to.biases[i] += velocity;
    }
}
