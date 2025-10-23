//
// Created by zacha on 20-10-25.
//

#include "neural_network.h"

#include <stdlib.h>

inline neural_network* alloc_neural_network(const int layerCount) {
    neural_network* result = malloc(sizeof(neural_network));
    result->layerCount = layerCount;
    result->layers = malloc(layerCount * sizeof(layer*));
    result->optimizer = NULL;

    return result;
}

inline void free_neural_network(neural_network* network, const int freeConstructed) {
    if (freeConstructed) {
        for (int l = 0; l < network->layerCount; l++) {
            layer* layer = network->layers[l];
            layer->functions.free(layer);
        }

        free(network->optimizer);
    }

    free(network->layers);
    free(network);
}

inline int get_in_count(const neural_network* network) {
    return network->layers[0]->in_count;
}

inline int get_out_count(const neural_network* network) {
    return network->layers[network->layerCount - 1]->out_count;
}

inline void set_cost_type(const neural_network* network) {

}

inline void predict(const neural_network* network, const double* inputs, double* outputs) {
    double* v = NULL;

    for (int i = 0; i < network->layerCount; i++) {
        const int isLast = i == network->layerCount - 1;
        const layer* l = network->layers[i];

        if (isLast) {
            l->functions.forward(l, v, outputs);
            free(v);
            return;
        }

        if (i == 0) {
            v = malloc(sizeof(double) * l->out_count);
            l->functions.forward(l, inputs, v);
        } else if (l->in_count != l->out_count) {
            double* temp = malloc(sizeof(double) * l->out_count);
            l->functions.forward(l, v, temp);
            free(v);
            v = temp;
        } else {
            l->functions.forward(l, v, v);
        }
    }
}

static double** alloc_layer_output_buffers(const neural_network* network) {
    double** result = malloc(sizeof(double*) * network->layerCount);

    for (int i = 0; i < network->layerCount; i++) {
        result[i] = malloc(sizeof(double) * network->layers[i]->out_count);
    }

    return result;
}

static void free_buffers(const neural_network* network, double** buffers) {
    for (int i = 0; i < network->layerCount; i++) {
        free(buffers[i]);
    }

    free(buffers);
}

static void average_gradients(const neural_network* network, double** buffers, const int count) {
    for (int i = 0; i < network->layerCount; i++) {
        for (int o = 0; o < network->layers[i]->gradient_count; o++) {
            buffers[i][o] /= count;
        }
    }
}

inline void learn(const neural_network* network, const test_data data, const range range, const optimizer_args args) {
    const int lastIndex = network->layerCount - 1;
    const int in_count = get_in_count(network);
    const int out_count = get_out_count(network);

    double** gradients = malloc(sizeof(double*) * network->layerCount);
    double** intermediateValues = alloc_layer_output_buffers(network);

    for (int r = range.from; r < range.to; r++) {
        const double* inputs = data.inputs + r * in_count;
        const double* expected = data.expected + r * out_count;

        //forward pass
        for (int i = 0; i < network->layerCount; i++) {
            const layer* l = network->layers[i];
            const double* in = i == 0 ? inputs : intermediateValues[i - 1];
            l->functions.forward(l, in, intermediateValues[i]);
        }

        double* currentDeltas = malloc(sizeof(double) * out_count);
        network->cost_vtable->get_cost_deltas(intermediateValues[lastIndex], expected, currentDeltas, out_count);

        //backward pass
        for (int i = lastIndex; i >= 0; i--) {
            const layer* l = network->layers[i];

            if (l->gradient_count > 0) {
                gradients[i] = malloc(sizeof(double) * l->gradient_count);
                const double* in = i == 0 ? inputs : intermediateValues[i - 1];
                l->functions.deltas_to_gradients(l, in, currentDeltas, gradients[i]);
            } else gradients[i] = NULL;

            if (i == 0) break;

            if (l->in_count == l->out_count) l->functions.backward(l, intermediateValues[i - 1], currentDeltas, currentDeltas);
            else {
                double* buffer = malloc(sizeof(double) * l->in_count);
                l->functions.backward(l, intermediateValues[i - 1], currentDeltas, buffer);
                free(currentDeltas);
                currentDeltas = buffer;
            }
        }
    }

    average_gradients(network, gradients, range.from - range.to);

    for (int i = 0; i < network->layerCount; i++) {
        const double* g = gradients[i];
        if (g == NULL) continue;

        const layer* l = network->layers[i];
        l->functions.apply_gradients(l, g, network->optimizer, args);
    }

    free_buffers(network, gradients);
    free_buffers(network, intermediateValues);
}

inline void initialize(const neural_network* network) {
    for (int i = 0; i < network->layerCount; i++) {
        layer* l = network->layers[i];
        l->functions.initialize(l);
    }
}

double get_cost(const neural_network* network, const double* inputs, const double* expected) {
    const int c = get_out_count(network);
    double* predicted = malloc(sizeof(double) * c);
    predict(network, inputs, predicted);

    const double cost = network->cost_vtable->get_cost(predicted, expected, c);

    free(predicted);
    return cost;
}

double get_avg_cost(const neural_network* network, test_data data) {
    const int in_count = get_in_count(network);
    const int out_count = get_out_count(network);
    double cost = 0;

    for (int i = 0; i < data.count; i++) {
        cost += get_cost(network, data.inputs + i * in_count, data.expected + i * out_count);
    }

    return cost / data.count;
}
