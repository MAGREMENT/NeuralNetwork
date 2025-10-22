//
// Created by zacha on 20-10-25.
//

#include "neural_network.h"

#include <stdlib.h>
#include <string.h>

inline neural_network* alloc_neural_network(int layerCount, layer** layers) {
    neural_network* result = malloc(sizeof(neural_network));
    result->layerCount = layerCount;
    result->layers = layers;

    return result;
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

static void set_buffers_to_zero(const neural_network* network, double** buffers) {
    for (int i = 0; i < network->layerCount; i++) {
        memset(buffers[i], 0, sizeof(double) * network->layers[i]->out_count);
    }
}

static void free_buffers(const neural_network* network, double** buffers) {
    for (int i = 0; i < network->layerCount; i++) {
        free(buffers[i]);
    }

    free(buffers);
}

static void average_deltas(const neural_network* network, double** buffers, const int count) {
    for (int i = 0; i < network->layerCount; i++) {
        for (int o = 0; o < network->layers[i]->out_count; o++) {
            buffers[i][o] /= count;
        }
    }
}

inline void learn(const neural_network* network, test_data data, range range, optimizer_args args) {
    const int lastIndex = network->layerCount - 1;
    const int in_count = network->layers[0]->in_count;
    const int out_count = network->layers[lastIndex]->out_count;

    double** deltas = alloc_layer_output_buffers(network);
    double** intermediateValues = alloc_layer_output_buffers(network);

    set_buffers_to_zero(network, deltas);

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
        network->get_cost_deltas(intermediateValues[lastIndex], expected, currentDeltas, out_count);

        //backward pass
        for (int i = lastIndex; i >= 0; i--) {
            const layer* l = network->layers[i];

            for (int o = 0; o < l->out_count; o++) {
                deltas[i][o] += currentDeltas[o];
            }

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

    //TODO look into ignoring non teachable layers
    average_deltas(network, deltas, range.from - range.to);

    //TODO need to separate gradients from deltas
    for (int i = 0; i < network->layerCount; i++) {
        const layer* l = network->layers[i];

        //l->functions.apply_gradients(l, )
    }

    free_buffers(network, deltas);
    free_buffers(network, intermediateValues);
}

inline void initialize(const neural_network* network) {
    for (int i = 0; i < network->layerCount; i++) {
        layer* l = network->layers[i];
        l->functions.initialize(l);
    }
}
