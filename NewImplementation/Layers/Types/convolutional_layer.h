//
// Created by zacha on 23-10-25.
//

#ifndef CONVOLUTIONNAL_LAYER_H
#define CONVOLUTIONNAL_LAYER_H

#include "../layer.h"

typedef struct size3D {
    int width;
    int height;
    int depth;
} size3D;

typedef struct conv_layer_params {
    size3D input_size;
    size3D kernel_size;
    size3D output_size;

    int stride;
    int padding;

    double* kernels;
    double* biases;
} conv_layer_params;

layer* cnstr_conv_layer(size3D inputSize, size3D kernelSize, int kernelCount, int stride, int padding);
void set_kernels_and_biases(const layer* l, double kernels, double biases);

#endif //CONVOLUTIONNAL_LAYER_H
