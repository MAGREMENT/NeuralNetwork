//
// Created by zacha on 23-10-25.
//

#include "convolutional_layer.h"

#include <stdlib.h>
#include <string.h>

#include "../../Util/math_util.h"

static void free_conv_layer(layer* l) {
    conv_layer_params* p = l->params;

    free(p->kernels);
    free(p->biases);
    free(p);
    free(l);
}

static void forward_conv_layer(const layer* l, const double* inputs, double* outputs) {
    const conv_layer_params* p = l->params;

    memcpy(outputs, p->biases, p->output_size.width * p->output_size.height * p->output_size.depth * sizeof(double));
    valid_correlate_add(inputs, p->input_size, p->kernels, p->kernel_size, outputs, p->output_size, p->padding, p->stride);
}

//TODO test
static void conv_backward(const layer* l, const double* inputs, const double* deltas, double* outputs) {
    const conv_layer_params* p = l->params;
    const int kSize = p->kernel_size.width * p->kernel_size.height * p->input_size.depth;

    memset(outputs, 0, sizeof(double) * p->input_size.width * p->input_size.height * p->input_size.depth);
    for (int d = 0; d < p->output_size.depth; d++) {
        full_convolve_add(deltas, p->output_size, p->kernels + kSize * d, p->kernel_size,
            outputs, p->input_size, p->padding, p->stride);
    }
}

//TODO test
static void conv_delta_to_gradients(const layer* l, const double* inputs, const double* deltas, double* gradients) {
    const conv_layer_params* p = l->params;

    const int oArea = p->output_size.width * p->output_size.height;

    for (int d = 0; d < p->output_size.depth; d++) {
        for (int w = 0; w < p->output_size.width; w++) {
            for (int h = 0; h < p->output_size.height; h++) {
                const int bIndex = d * oArea + h * p->output_size.width + w;
                gradients[bIndex] = deltas[bIndex];
            }
        }
    }

    valid_correlate_add(inputs, p->input_size, deltas, to2D(p->output_size), gradients,
        to3D(p->kernel_size, p->output_size.depth), p->padding, p->stride);
}

static void apply_gradients_to_conv(const layer* l, const double* gradients, const optimizer* opt, optimizer_args args) {
    const conv_layer_params* p = l->params;

    const int kFullSize = p->kernel_size.width * p->kernel_size.height * p->input_size.depth * p->output_size.depth;
    const int oSize = p->output_size.width * p->output_size.height * p->output_size.depth;

    opt->vtable->apply_gradients(opt, p->kernels, gradients, kFullSize, args);
    opt->vtable->apply_gradients(opt, p->biases, gradients + kFullSize, oSize, args);
}

layer_vtable conv_vtable = {forward_conv_layer, conv_backward, conv_delta_to_gradients, apply_gradients_to_conv, free_conv_layer};

layer* cnstr_conv_layer(const size3D inputSize, const size2D kernelSize, const int kernelCount, const int stride, const int padding) {
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

void set_kernels_and_biases(const layer* l, const double kernels, const double biases) {
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