#include <stdio.h>
#include "repository.h"

struct neural_network* example_network();

int main() {
    struct neural_network* network = example_network();

    struct input_data *data = alloc_input_data(2);
    double v[] = {1, 2};
    set_input_data(*data, v);

    struct input_data* d1 = forward(network->layers[0], *data, network->activation);
    struct input_data* d2 = forward(network->layers[1], *d1, network->activation);

    printf("%f %f\n", data->values[0], data->values[1]);
    printf("%f %f %f\n", d1->values[0], d1->values[1], d1->values[2]);
    printf("%f %f", d2->values[0], d2->values[1]);

    free_network(network);
    return EXIT_SUCCESS;
}

struct neural_network* example_network(){
    int numbers[] = {2, 3, 2};
    struct neural_network* network = alloc_network(3, numbers);

    struct params params;
    params.learningRate = 0.8;
    params.activationType = SIGMOID;
    params.costType = MEAN_SQUARED;
    apply_params(network, params);

    double w1[] = {0.5, -7, 2, -1, 2, 3};
    double b1[] = {1, 2, -3};
    set_layer(network->layers[0], w1, b1);

    double w2[] = {-2, 3, -4, 3, 0.7, 8};
    double b2[] = {0, 1};
    set_layer(network->layers[1], w2, b2);

    return network;
}

