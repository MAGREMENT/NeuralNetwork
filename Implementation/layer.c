//
// Created by zacha on 01-10-25.
//

#include "layer.h"

#include <stdlib.h>

inline void free_layer_data_array(layer_data* gradients, const int count) {
    for(int i = 0; i < count; i++){
        free(gradients[i].biases);
        free(gradients[i].weights);
    }

    free(gradients);
}
