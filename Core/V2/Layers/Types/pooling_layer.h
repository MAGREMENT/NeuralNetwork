//
// Created by zacha on 28-10-25.
//

#ifndef NEWIMPLEMENTATION_POOLING_LAYER_H
#define NEWIMPLEMENTATION_POOLING_LAYER_H

#include "../layer.h"
#include "../../Util/size.h"

enum pooling_types {
    POOLING_MAX,
    POOLING_AVG,
    POOLING_LP,
    POOLING_STOCHASTIC
};

typedef struct pooling_layer_params {
    size3D input_size;
    size2D window_size;
    size3D output_size;

    int stride;
    int padding;
} pooling_layer_params;

extern layer* cnstr_pooling_layer(int type, size3D inputSize, size2D windowSize, int stride, int padding);

#endif //NEWIMPLEMENTATION_POOLING_LAYER_H