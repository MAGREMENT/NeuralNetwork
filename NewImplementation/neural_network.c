//
// Created by zacha on 20-10-25.
//

#include "neural_network.h"

#include <stdlib.h>

inline neural_network* alloc_neural_network(int layerCount, layer** layers) {
    neural_network* result = malloc(sizeof(neural_network));
    result->layerCount = layerCount;
    result->layers = layers;

    return result;
}

inline double* predict(neural_network* network, double* inputs) {
    double* v = inputs;
    int isNoLongerInputs = false;
    for (int i = 0; i < network->layerCount; i++) {
        const layer* l = network->layers[i];
        int didAllocate = false;
        double* buffer = l->functions.forward(l, v, &didAllocate);

        if (didAllocate) {
            if (isNoLongerInputs) free(v);
            else isNoLongerInputs = true;
        }

        v = buffer;
    }

    return v;
}

inline void learn(neural_network* network, const double* input, const double* expected, const int count) {
    double* deltas = malloc(sizeof(double) * count);
    network->get_cost_deltas(input, expected, deltas, count);

    //TODO forward pass first

    int didAllocate;
    for (int i = network->layerCount - 1; i >= 0; i--) {
        const layer* l = network->layers[i];
        double* buffer = l->functions.backward(l, input, deltas, &didAllocate);
    }
}

inline void initialize(neural_network* network) {
    for (int i = 0; i < network->layerCount; i++) {
        layer* l = network->layers[i];
        l->functions.initialize(l);
    }
}
