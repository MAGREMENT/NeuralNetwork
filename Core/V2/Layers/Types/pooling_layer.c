//
// Created by zacha on 28-10-25.
//

#include "pooling_layer.h"

#include <float.h>
#include <stdlib.h>
#include <string.h>

static void forward_max_pooling_layer(const layer* l, const double* inputs, double* outputs) {
    const pooling_layer_params* p = l->data;

    const int inArea = p->input_size.width * p->input_size.height;
    const int outArea = p->output_size.width * p->output_size.height;

    for (int d = 0; d < p->output_size.depth; d++) {
        for (int oW = 0; oW < p->output_size.width; oW++) {
            for (int oH = 0; oH < p->output_size.height; oH++) {
                const int w = oW * p->stride - p->padding;
                const int h = oH * p->stride - p->padding;

                double result = -DBL_MAX;

                for (int kW = 0; kW < p->window_size.width; kW++) {
                    for (int kH = 0; kH < p->window_size.height; kH++) {
                        const int currW = w + kW;
                        const int currH = h + kH;

                        if (currW < 0 || currW >= p->input_size.width ||
                            currH < 0 || currH >= p->input_size.height) continue;

                        const int inIndex = inArea * d + currH * p->input_size.width + currW;
                        if (inputs[inIndex] > result) result = inputs[inIndex];
                    }
                }

                outputs[outArea * d + oH * p->output_size.width + oW] = result;
            }
        }
    }
}

static void forward_avg_pooling_layer(const layer* l, const double* inputs, double* outputs) {
    const pooling_layer_params* p = l->data;

    const int inArea = p->input_size.width * p->input_size.height;
    const int outArea = p->output_size.width * p->output_size.height;
    const double div = p->window_size.width * p->window_size.height;

    for (int d = 0; d < p->output_size.depth; d++) {
        for (int oW = 0; oW < p->output_size.width; oW++) {
            for (int oH = 0; oH < p->output_size.height; oH++) {
                const int w = oW * p->stride - p->padding;
                const int h = oH * p->stride - p->padding;

                double result = 0;

                for (int kW = 0; kW < p->window_size.width; kW++) {
                    for (int kH = 0; kH < p->window_size.height; kH++) {
                        const int currW = w + kW;
                        const int currH = h + kH;

                        if (currW < 0 || currW >= p->input_size.width ||
                            currH < 0 || currH >= p->input_size.height) continue;

                        const int inIndex = inArea * d + currH * p->input_size.width + currW;
                        result += inputs[inIndex];
                    }
                }

                outputs[outArea * d + oH * p->output_size.width + oW] = result / div;
            }
        }
    }
}

static void backward_max_pooling_layer(const layer* l, const double* inputs, const double* deltas, double* outputs) {
    const pooling_layer_params* p = l->data;
    const int inArea = p->input_size.width * p->input_size.height;
    const int outArea = p->output_size.width * p->output_size.height;

    for (int d = 0; d < p->output_size.depth; d++) {
        for (int oW = 0; oW < p->output_size.width; oW++) {
            for (int oH = 0; oH < p->output_size.height; oH++) {
                const int w = oW * p->stride - p->padding;
                const int h = oH * p->stride - p->padding;

                double max = -DBL_MAX;
                int ind = -1;

                for (int kW = 0; kW < p->window_size.width; kW++) {
                    for (int kH = 0; kH < p->window_size.height; kH++) {
                        const int currW = w + kW;
                        const int currH = h + kH;

                        if (currW < 0 || currW >= p->input_size.width ||
                            currH < 0 || currH >= p->input_size.height) continue;

                        const int inIndex = inArea * d + currH * p->input_size.width + currW;
                        outputs[inIndex] = 0;

                        if (inputs[inIndex] > max) {
                            max = inputs[inIndex];
                            ind = inIndex;
                        }
                    }
                }

                if (ind != -1) outputs[ind] += deltas[d * outArea + oH * p->output_size.width + oW];
            }
        }
    }
}

static void backward_avg_pooling_layer(const layer* l, const double* inputs, const double* deltas, double* outputs) {
    const pooling_layer_params* p = l->data;

    const int inArea = p->input_size.width * p->input_size.height;
    const int outArea = p->output_size.width * p->output_size.height;
    const double div = p->window_size.width * p->window_size.height;

    memset(outputs, 0, inArea * p->input_size.depth * sizeof(double));

    for (int d = 0; d < p->output_size.depth; d++) {
        for (int oW = 0; oW < p->output_size.width; oW++) {
            for (int oH = 0; oH < p->output_size.height; oH++) {
                const int w = oW * p->stride - p->padding;
                const int h = oH * p->stride - p->padding;

                const double v = deltas[d * outArea + oH * p->output_size.width + oW] / div;

                for (int kW = 0; kW < p->window_size.width; kW++) {
                    for (int kH = 0; kH < p->window_size.height; kH++) {
                        const int currW = w + kW;
                        const int currH = h + kH;

                        if (currW < 0 || currW >= p->input_size.width ||
                            currH < 0 || currH >= p->input_size.height) continue;

                        const int inIndex = inArea * d + currH * p->input_size.width + currW;
                        outputs[inIndex] += v;
                    }
                }
            }
        }
    }
}

layer_vtable pooling_vtable_store[] = {
    {forward_max_pooling_layer, backward_max_pooling_layer, no_delta_to_gradients, apply_no_gradients, default_layer_free},
    {forward_avg_pooling_layer, backward_avg_pooling_layer, no_delta_to_gradients, apply_no_gradients, default_layer_free}
};

layer* cnstr_pooling_layer(const int type, const size3D inputSize, const size2D windowSize, const int stride, const int padding) {
    layer* l = malloc(sizeof(layer));
    pooling_layer_params* p = malloc(sizeof(pooling_layer_params));

    p->input_size = inputSize;
    p->window_size = windowSize;
    p->stride = stride;
    p->padding = padding;

    p->output_size.depth = inputSize.depth;
    p->output_size.width = (inputSize.width - windowSize.width + 2 * padding) / stride + 1;
    p->output_size.height = (inputSize.height - windowSize.height + 2 * padding) / stride + 1;

    l->in_count = inputSize.width * inputSize.height * inputSize.depth;
    l->out_count = p->output_size.width * p->output_size.height * p->output_size.depth;

    l->parameters_count = 0;
    l->parameters = NULL;

    l->data = p;

    l->vtable = pooling_vtable_store + type;

    return l;
}
