//
// Created by zacha on 23-10-25.
//

#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "../layer.h"

#include "../../Util/size.h"

typedef struct conv_layer_params {
    size3D input_size;
    size2D kernel_size;
    size3D output_size;

    int stride;
    int padding;

    double* kernels;
    double* biases;
} conv_layer_params;

extern layer* cnstr_conv_layer(size3D inputSize, size2D kernelSize, int kernelCount, int stride, int padding);
extern void set_kernels_and_biases(const layer* l, double kernels, double biases);

#endif //CONVOLUTIONAL_LAYER_H
