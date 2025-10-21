//
// Created by zacha on 20-10-25.
//

#include "layer.h"

#include <stdlib.h>

inline void no_initialization(layer* l) {}

inline void default_layer_free(layer* l) {
    free(l->params);
    free(l);
}