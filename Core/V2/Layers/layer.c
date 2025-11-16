//
// Created by zacha on 20-10-25.
//

#include "layer.h"

#include <stdlib.h>

inline void no_initialization(const layer* l) {}

inline void default_layer_free(layer* l) {
    free(l->data);
    free(l->parameters);
    free(l);
}

inline void no_delta_to_gradients(const layer* l, const double* inputs, const double* deltas, double* gradients) {}