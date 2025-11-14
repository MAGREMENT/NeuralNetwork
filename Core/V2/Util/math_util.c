//
// Created by zacha on 22-10-25.
//

#include "math_util.h"

#include <math.h>
#include "../asserter.h"

inline double sigmoid(const double input) {
    return 1 / (1 + exp(-input));
}

//TODO test
void valid_correlate_add(const double* inputs, const size3D input_size, const double* kernels, const size2D kernel_size,
    double* outputs, const size3D output_size, const int padding, const int stride) {

    assert((input_size.width - kernel_size.width + 2 * padding) / stride + 1 == output_size.width);
    assert((input_size.height - kernel_size.height + 2 * padding) / stride + 1 == output_size.height);

    const int oArea = output_size.width * output_size.height;
    const int kernelArea = kernel_size.width * kernel_size.height;
    const int inputArea = input_size.width * input_size.height;

    for (int c = 0; c < output_size.depth; c++) {
        for (int oW = 0; oW < output_size.width; oW++) {
            for (int oH = 0; oH < output_size.height; oH++) {
                const int w = oW * stride - padding;
                const int h = oH * stride - padding;

                const int oIndex = c * oArea + oH * output_size.width + oW;
                double result = 0;

                for (int kW = 0; kW < kernel_size.width; kW++) {
                    for (int kH = 0; kH < kernel_size.height; kH++) {
                        const int currW = w + kW;
                        const int currH = h + kH;

                        if (currW < 0 || currW >= input_size.width ||
                            currH < 0 || currH >= input_size.height) continue;

                        for (int d = 0; d < input_size.depth; d++) {
                            const int inIndex = inputArea * d + currH * input_size.width + currW;
                            const int kernIndex = kernelArea * d + kH * kernel_size.width + kW;

                            result += inputs[inIndex] * kernels[kernIndex];
                        }
                    }
                }

                outputs[oIndex] += result;
            }
        }
    }
}

//TODO test
void full_convolve_add(const double* inputs, const size3D input_size, const double* kernels, const size2D kernel_size,
    double* outputs, const size3D output_size, const int padding, const int stride) {

    assert((input_size.width - 1) * stride - 2 * padding + kernel_size.width == output_size.width);
    assert((input_size.height - 1) * stride - 2 * padding + kernel_size.height == output_size.height);

    const int oArea = output_size.width * output_size.height;
    const int kernelArea = kernel_size.width * kernel_size.height;
    const int inputArea = input_size.width * input_size.height;

    for (int c = 0; c < output_size.depth; c++) {
        for (int oW = 0; oW < output_size.width; oW++) {
            for (int oH = 0; oH < output_size.height; oH++) {
                const int w = oW * stride - padding - kernel_size.width + 1;
                const int h = oH * stride - padding - kernel_size.height + 1;

                const int oIndex = c * oArea + oH * output_size.width + oW;
                double result = 0;

                for (int kW = 0; kW < kernel_size.width; kW++) {
                    for (int kH = 0; kH < kernel_size.height; kH++) {
                        const int currW = w + kW;
                        const int currH = h + kH;

                        if (currW < 0 || currW >= input_size.width ||
                            currH < 0 || currH >= input_size.height) continue;

                        for (int d = 0; d < input_size.depth; d++) {
                            const int inIndex = inputArea * d + currH * input_size.width + currW;
                            const int kernIndex = kernelArea * d + (kernel_size.height - kH) * kernel_size.width + (kernel_size.width - kW);

                            result += inputs[inIndex] * kernels[kernIndex];
                        }
                    }
                }

                outputs[oIndex] += result;
            }
        }
    }
}
