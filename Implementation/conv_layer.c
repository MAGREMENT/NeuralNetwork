//
// Created by zacha on 17-10-25.
//

#include "conv_layer.h"

#include <stdlib.h>

inline void get_output_size(conv_layer* layer, int* width_result, int* height_result, int* depth_result) {
    *depth_result = layer->kernel_count;
    *width_result = (layer->input_size.width - layer->kernel_size.width + 2 * layer->padding) / layer->stride + 1;
    *height_result = (layer->input_size.height - layer->kernel_size.height + 2 * layer->padding) / layer->stride + 1;
}

static double* alloc_output(conv_layer* l) {
    int bWidth;
    int bHeight;
    int bDepth;

    get_output_size(l, &bWidth, &bHeight, &bDepth);
    const int bSize = bWidth * bHeight * bDepth;
    return malloc(sizeof(double) * bSize);
}

inline conv_layer* alloc_conv_layer(size3D inputSize, size3D kernelSize, int kernelCount, int stride, int padding) {
    conv_layer* l = malloc(sizeof(conv_layer));
    l->kernel_size = kernelSize;
    l->input_size = inputSize;
    l->kernel_count = kernelCount;
    l->stride = stride;
    l->padding = padding;

    const int kSize = kernelSize.width * kernelSize.height * kernelSize.depth * kernelCount;
    l->kernels = malloc(sizeof(double) * kSize);

    l->biases = alloc_output(l);

    return l;
}

inline void free_conv_layer(conv_layer* layer) {
    free(layer->kernels);
    free(layer->biases);
    free(layer);
}

double* conv_forward(conv_layer* l, const double* input) {
    int bWidth;
    int bHeight;
    int bDepth;

    get_output_size(l, &bWidth, &bHeight, &bDepth);
    const int bArea = bWidth * bHeight;
    double* o = malloc(sizeof(double) * bArea * bDepth);

    const int kernelArea = l->kernel_size.width * l->kernel_size.height;
    const int inputArea = l->input_size.width * l->input_size.height;

    for (int c = 0; c < l->kernel_count; c++) {
        for (int oW = 0; oW < bWidth; oW++) {
            for (int oH = 0; oH < bHeight; oH++) {
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

                const int oIndex = c * bArea + oH * bWidth + oW;
                o[oIndex] = result + l->biases[oIndex];
            }
        }
    }

    return o;
}