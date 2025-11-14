//
// Created by zacha on 20-10-25.
//

#include "layer.h"

#include <stdlib.h>

inline void no_initialization(const layer* l) {}

inline void default_layer_free(layer* l) {
    free(l->params);
    free(l);
}

inline void no_delta_to_gradients(const layer* l, const double* inputs, const double* deltas, double* gradients) {}

inline void apply_no_gradients(const layer* l, const double* gradients, const optimizer* opt, optimizer_args args) {}

inline void no_export(const layer* l, double* parameters) {}

inline void no_import(const layer* l, const double* parameters) {}