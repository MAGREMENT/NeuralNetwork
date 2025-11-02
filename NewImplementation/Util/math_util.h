//
// Created by zacha on 22-10-25.
//

#ifndef STRING_MATH_H
#define STRING_MATH_H
#include "size.h"

double sigmoid(double input);
void valid_correlate_add(const double* inputs, size3D input_size, const double* kernels, size2D kernel_size,
    double* outputs, size3D output_size, int padding, int stride);
void full_convolve_add(const double* inputs, size3D input_size, const double* kernels, size2D kernel_size,
    double* outputs, size3D output_size, int padding, int stride);

#endif //STRING_MATH_H
