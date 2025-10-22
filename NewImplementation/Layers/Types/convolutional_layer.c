//
// Created by zacha on 23-10-25.
//

#include "convolutional_layer.h"

#include <stdlib.h>

static void free_conv_layer(layer* l) {
    conv_layer_params* p = l->params;

    free(p->kernels);
    free(p->biases);
    free(p);
    free(l);
}

static void forward_conv_layer(const layer* l, const double* inputs, double* outputs) {
    const conv_layer_params* p = l->params;

    const int oArea = p->output_size.width * p->output_size.height;
    const int kernelArea = p->kernel_size.width * p->kernel_size.height;
    const int inputArea = p->input_size.width * p->input_size.height;

    for (int c = 0; c < p->output_size.depth; c++) {
        for (int oW = 0; oW < p->output_size.width; oW++) {
            for (int oH = 0; oH < p->output_size.height; oH++) {
                const int w = oW * p->stride - p->padding;
                const int h = oH * p->stride - p->padding;
                double result = 0;

                for (int kW = 0; kW < p->kernel_size.width; kW++) {
                    for (int kH = 0; kH < p->kernel_size.height; kH++) {
                        const int currW = w + kW;
                        const int currH = h + kH;

                        if (currW < 0 || currW >= p->input_size.width ||
                            currH < 0 || currH >= p->input_size.height) continue;

                        for (int d = 0; d < p->input_size.depth; d++) {
                            const int inIndex = inputArea * d + currH * p->input_size.width + currW;
                            const int kernIndex = kernelArea * d + kH * p->kernel_size.height + kW;

                            result += inputs[inIndex] * p->kernels[kernIndex];
                        }
                    }
                }

                const int oIndex = c * oArea + oH * p->output_size.width + oW;
                outputs[oIndex] = result + p->biases[oIndex];
            }
        }
    }
}

inline layer* cnstr_conv_layer(size3D inputSize, size3D kernelSize, int kernelCount, int stride, int padding) {
    conv_layer_params* p = malloc(sizeof(conv_layer_params));
    p->kernel_size = kernelSize;
    p->input_size = inputSize;
    p->stride = stride;
    p->padding = padding;

    p->output_size.depth = kernelCount;
    p->output_size.width = (inputSize.width - kernelSize.width + 2 * padding) / stride + 1;
    p->output_size.height = (inputSize.height - kernelSize.height + 2 * padding) / stride + 1;

    const int kSize = kernelSize.width * kernelSize.height * kernelSize.depth * kernelCount;
    p->kernels = malloc(sizeof(double) * kSize);

    const int bSize = p->output_size.width * p->output_size.height * p->output_size.depth;
    p->biases = malloc(sizeof(double) * bSize);

    layer* l = malloc(sizeof(layer));

    l->params = p;
    l->in_count = inputSize.width * inputSize.height * inputSize.depth;
    l->out_count = p->output_size.depth * p->output_size.width * p->output_size.height;
    l->gradient_count = kSize + bSize;

    l->functions.free = free_conv_layer;
    l->functions.forward = forward_conv_layer;

    return l;
}

inline void set_kernels_and_biases(const layer* l, const double kernels, const double biases) {
    const conv_layer_params* p = l->params;

    size_t size = p->kernel_size.width * p->kernel_size.height * p->kernel_size.depth;
    for (size_t i = 0; i < size; i++) {
        p->kernels[i] = kernels;
    }

    size = l->out_count;
    for (size_t i = 0; i < size; i++) {
        p->biases[i] = biases;
    }
}