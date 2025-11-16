//
// Created by zacha on 23-10-25.
//

#include "convolutional_layer.h"

#include <stdlib.h>
#include <string.h>
#include <tgmath.h>

#include "../../Util/math_util.h"
#include "../../Util/rand_util.h"

static double* get_biases(const layer* l, const conv_layer_params* p) {
    return l->parameters + p->kernel_volume;
}

static void forward_conv_layer(const layer* l, const double* inputs, double* outputs) {
    const conv_layer_params* p = l->data;

    memcpy(outputs, get_biases(l, p), p->output_size.width * p->output_size.height * p->output_size.depth * sizeof(double));
    valid_correlate_add(inputs, p->input_size, l->parameters, p->kernel_size, outputs, p->output_size, p->padding, p->stride);
}

//TODO test
static void conv_backward(const layer* l, const double* inputs, const double* deltas, double* outputs) {
    const conv_layer_params* p = l->data;
    const int kSize = p->kernel_size.width * p->kernel_size.height * p->input_size.depth;

    memset(outputs, 0, sizeof(double) * p->input_size.width * p->input_size.height * p->input_size.depth);
    for (int d = 0; d < p->output_size.depth; d++) {
        full_convolve_add(deltas, p->output_size, l->parameters + kSize * d, p->kernel_size,
            outputs, p->input_size, p->padding, p->stride);
    }
}

//TODO test
static void conv_delta_to_gradients(const layer* l, const double* inputs, const double* deltas, double* gradients) {
    const conv_layer_params* p = l->data;

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

layer_vtable conv_vtable = {forward_conv_layer, NULL, conv_backward, conv_delta_to_gradients, default_layer_free};

layer* cnstr_conv_layer(const size3D inputSize, const size2D kernelSize, const int kernelCount, const int stride, const int padding, void (*initialize)(const layer* l)) {
    conv_layer_params* p = malloc(sizeof(conv_layer_params));
    p->kernel_size = kernelSize;
    p->input_size = inputSize;
    p->stride = stride;
    p->padding = padding;

    p->output_size.depth = kernelCount;
    p->output_size.width = (inputSize.width - kernelSize.width + 2 * padding) / stride + 1;
    p->output_size.height = (inputSize.height - kernelSize.height + 2 * padding) / stride + 1;

    p->kernel_volume = kernelSize.width * kernelSize.height * inputSize.depth * kernelCount;
    const int bSize = p->output_size.width * p->output_size.height * p->output_size.depth;

    layer* l = malloc(sizeof(layer));

    l->data = p;

    l->in_count = inputSize.width * inputSize.height * inputSize.depth;
    l->out_count = p->output_size.depth * p->output_size.width * p->output_size.height;

    l->parameters_count = p->kernel_volume + bSize;
    l->parameters = malloc(sizeof(double) * l->parameters_count);

    l->initialize = initialize;
    l->vtable = &conv_vtable;

    return l;
}

void set_kernels_and_biases(const layer* l, const double kernels, const double biases) {
    const conv_layer_params* p = l->data;
    double* b = get_biases(l, p);

    for (int i = 0; i < p->kernel_volume; i++) {
        l->parameters[i] = kernels;
    }

    for (int i = 0; i < l->out_count; i++) {
        b[i] = biases;
    }
}

void initialize_conv_random(const layer* layer) {
    const conv_layer_params* p = layer->data;

    for (int i = 0; i < p->kernel_volume; i++) {
        layer->parameters[i] = rand_d_std_nrml_distr() * 0.01;
    }

    memset(get_biases(layer, p), 0, sizeof(double) * layer->out_count);
}

void initialize_conv_he(const layer* layer) {
    const conv_layer_params* p = layer->data;

    const double scale = sqrt(2.0 / (p->kernel_size.width * p->kernel_size.height * p->input_size.depth));
    for (int i = 0; i < p->kernel_volume; i++) {
        layer->parameters[i] = rand_d_std_nrml_distr() * scale;
    }

    memset(get_biases(layer, p), 0, sizeof(double) * layer->out_count);
}

void initialize_conv_xavier(const layer* layer) {
    const conv_layer_params* p = layer->data;

    const double scale = sqrt(1.0 / (p->kernel_size.width * p->kernel_size.height * p->input_size.depth));
    for (int i = 0; i < p->kernel_volume; i++) {
        layer->parameters[i] = rand_d_std_nrml_distr() * scale;
    }

    memset(get_biases(layer, p), 0, sizeof(double) * layer->out_count);
}