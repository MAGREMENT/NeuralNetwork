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
    int kernel_count;

    int stride;
    int padding;

    double* kernels;
    double* biases;
} conv_layer;

void get_output_size(conv_layer* layer, int* width_result, int* height_result, int* depth_result);
conv_layer* alloc_conv_layer(size3D inputSize, size3D kernelSize, int kernelCount, int stride, int padding);
void free_conv_layer(conv_layer* layer);
double* conv_forward(conv_layer* l, const double* input);

#endif //CONVOLUTION_LAYER_H
