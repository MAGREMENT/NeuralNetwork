//
// Created by zacha on 17-10-25.
//

#ifndef CONVOLUTION_LAYER_H
#define CONVOLUTION_LAYER_H

typedef struct size3D {
    int width;
    int height;
    int depth;
} size3D;

typedef struct conv_layer {
    size3D input_size;
    size3D kernel_size;
    size3D output_size;

    int stride;
    int padding;

    double* kernels;
    double* biases;
} conv_layer;

conv_layer* alloc_conv_layer(size3D inputSize, size3D kernelSize, int kernelCount, int stride, int padding);
void free_conv_layer(conv_layer* layer);
double* conv_forward(conv_layer* l, const double* input);
void set_kernels_and_biases(conv_layer* l, double kernels, double biases);

#endif //CONVOLUTION_LAYER_H
