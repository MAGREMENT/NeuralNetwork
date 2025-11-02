//
// Created by zacha on 23-10-25.
//

#include "convolutional_layer.h"

#include <stdlib.h>
#include <string.h>

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

                const int oIndex = c * oArea + oH * p->output_size.width + oW;
                double result = p->biases[oIndex];

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

                outputs[oIndex] = result;
            }
        }
    }
}

//TODO look into the impact of kernel count
static void conv_delta_to_gradients(const layer* l, const double* inputs, const double* deltas, double* gradients) {
    const conv_layer_params* p = l->params;

    const int oArea = p->output_size.width * p->output_size.height;
    const int kArea = p->kernel_size.width * p->kernel_size.height;

    int bufferWidth, bufferHeight;
    if (p->padding == 0) {
        bufferWidth = p->input_size.width;
        bufferHeight = p->input_size.height;
    }
    else {
        bufferWidth = p->input_size.width + p->padding * 2;
        bufferHeight = p->input_size.height + p->padding * 2;
    }

    const int bufferArea = bufferWidth * bufferHeight;
    const int bufferSize = p->input_size.depth * bufferArea;

    double* buffer = p->padding == 0 ? gradients : malloc(bufferSize * sizeof(double));

    memset(buffer, 0, bufferSize * sizeof(double));

    for (int d = 0; d < p->output_size.depth; d++) {
        for (int w = 0; w < p->output_size.width; w++) {
            for (int h = 0; h < p->output_size.height; h++) {
                const int bIndex = d * oArea + h * p->output_size.width + w;
                gradients[bIndex] = deltas[bIndex];

                for (int kW = 0; kW < p->kernel_size.width; kW++) {
                    for (int kH = 0; kH < p->kernel_size.height; kH++) {
                        const int oIndex = d * bufferArea + kH * bufferWidth + kW + w * p->stride + h * p->stride * bufferWidth;
                        const int kIndex = d * kArea + kH * p->kernel_size.height + kW;
                        buffer[oIndex] += deltas[bIndex] * p->kernels[kIndex];
                    }
                }
            }
        }
    }

    if (p->padding != 0) {
        const int inArea = p->input_size.width * p->input_size.height;

        for (int d = 0; d < p->input_size.depth; d++) {
            for (int w = 0; w < p->input_size.width; w++) {
                for (int h = 0; h < p->input_size.height; h++) {
                    const int bIndex = d * bufferArea + h * bufferWidth + w + p->padding + p->padding * bufferWidth;
                    const int oIndex = d * inArea + h * p->input_size.width + w;
                    gradients[oIndex] = buffer[bIndex];
                }
            }
        }

        free(buffer);
    }
}

layer_vtable conv_vtable = {.forward = forward_conv_layer, .deltas_to_gradients = conv_delta_to_gradients, .free = free_conv_layer};

inline layer* cnstr_conv_layer(const size3D inputSize, const size2D kernelSize, const int kernelCount, const int stride, const int padding) {
    conv_layer_params* p = malloc(sizeof(conv_layer_params));
    p->kernel_size = kernelSize;
    p->input_size = inputSize;
    p->stride = stride;
    p->padding = padding;

    p->output_size.depth = kernelCount;
    p->output_size.width = (inputSize.width - kernelSize.width + 2 * padding) / stride + 1;
    p->output_size.height = (inputSize.height - kernelSize.height + 2 * padding) / stride + 1;

    const int kSize = kernelSize.width * kernelSize.height * inputSize.depth * kernelCount;
    p->kernels = malloc(sizeof(double) * kSize);

    const int bSize = p->output_size.width * p->output_size.height * p->output_size.depth;
    p->biases = malloc(sizeof(double) * bSize);

    layer* l = malloc(sizeof(layer));

    l->params = p;
    l->in_count = inputSize.width * inputSize.height * inputSize.depth;
    l->out_count = p->output_size.depth * p->output_size.width * p->output_size.height;
    l->gradient_count = kSize + bSize;

    l->vtable = &conv_vtable;

    return l;
}

inline void set_kernels_and_biases(const layer* l, const double kernels, const double biases) {
    const conv_layer_params* p = l->params;

    size_t size = p->kernel_size.width * p->kernel_size.height * p->input_size.depth;
    for (size_t i = 0; i < size; i++) {
        p->kernels[i] = kernels;
    }

    size = l->out_count;
    for (size_t i = 0; i < size; i++) {
        p->biases[i] = biases;
    }
}