//
// Created by zacha on 01-10-25.
//

#include "layer.h"

#include <stdlib.h>

inline layer_data* alloc_layer_data_array(layer* layers, const int layerCount, const int copyValues) {
    layer_data* result = malloc(layerCount * sizeof(layer_data));

    for(int n = 0; n < layerCount; n++) {
        const int in = layers[n].in_count;
        const int out = layers[n].out_count;

        result[n].biases = malloc(out * sizeof(double));
        result[n].weights = malloc(in * out * sizeof(double));

        for(int o = 0; o < out; o++) {
            for(int i = 0; i < in; i++) {
                result[n].weights[i * out + o] = copyValues ? layers[n].weights[i * out + o] : 0;
            }

            result[n].biases[o] = copyValues ? layers[n].biases[0] : 0;
        }
    }

    return result;
}

inline void free_layer_data_array(layer_data* layers, const int count) {
    for(int i = 0; i < count; i++){
        free(layers[i].biases);
        free(layers[i].weights);
    }

    free(layers);
}
