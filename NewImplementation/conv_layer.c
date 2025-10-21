//
// Created by zacha on 17-10-25.
//

#include "conv_layer.h"

#include <stdlib.h>
#include <string.h>

inline conv_layer* alloc_conv_layer(size3D inputSize, size3D kernelSize, int kernelCount, int stride, int padding) {
    conv_layer* l = malloc(sizeof(conv_layer));
    l->kernel_size = kernelSize;
    l->input_size = inputSize;
    l->stride = stride;
    l->padding = padding;

    l->output_size.depth = kernelCount;
    l->output_size.width = (inputSize.width - kernelSize.width + 2 * padding) / stride + 1;
    l->output_size.height = (inputSize.height - kernelSize.height + 2 * padding) / stride + 1;

    const int kSize = kernelSize.width * kernelSize.height * kernelSize.depth * kernelCount;
    l->kernels = malloc(sizeof(double) * kSize);

    const int bSize = l->output_size.width * l->output_size.height * l->output_size.depth;
    l->biases = malloc(sizeof(double) * bSize);

    return l;
}

inline void free_conv_layer(conv_layer* layer) {
    free(layer->kernels);
    free(layer->biases);
    free(layer);
}

double* conv_forward(conv_layer* l, const double* input) {
    const int bArea = l->output_size.width * l->output_size.height;
    double* o = malloc(sizeof(double) * 4);

    const int kernelArea = l->kernel_size.width * l->kernel_size.height;
    const int inputArea = l->input_size.width * l->input_size.height;

    for (int c = 0; c < l->output_size.depth; c++) {
        for (int oW = 0; oW < l->output_size.width; oW++) {
            for (int oH = 0; oH < l->output_size.height; oH++) {
                const int w = oW * l->stride - l->padding;
                const int h = oH * l->stride - l->padding;
                double result = 0;

                for (int kW = 0; kW < l->kernel_size.width; kW++) {
                    for (int kH = 0; kH < l->kernel_size.height; kH++) {
                        const int currW = w + kW;
                        const int currH = h + kH;

                        if (currW < 0 || currW >= l->input_size.width ||
                            currH < 0 || currH >= l->input_size.height) continue;

                        for (int d = 0; d < l->input_size.depth; d++) {
                            const int inIndex = inputArea * d + currH * l->input_size.width + currW;
                            const int kernIndex = kernelArea * d + kH * l->kernel_size.height + kW;

                            result += input[inIndex] * l->kernels[kernIndex];
                        }
                    }
                }

                const int oIndex = c * bArea + oH * l->output_size.width + oW;
                o[oIndex] = result + l->biases[oIndex];
            }
        }
    }

    return o;
}

inline void set_kernels_and_biases(conv_layer* l, double kernels, double biases) {
    size_t size = l->kernel_size.width * l->kernel_size.height * l->kernel_size.depth;
    for (size_t i = 0; i < size; i++) {
        l->kernels[i] = kernels;
    }

    size = l->output_size.width * l->output_size.height * l->output_size.depth;
    for (size_t i = 0; i < size; i++) {
        l->biases[i] = biases;
    }
}