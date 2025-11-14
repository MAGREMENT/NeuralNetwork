//
// Created by zacha on 09-11-25.
//

#include "i_o.h"

#include <math.h>
#include <string.h>

void generate_binary_inputs(double* result, const int max) {
    memset(result, 0, sizeof(double) * max);
    result[max] = 1;
    memset(result + max + 1, 0, sizeof(double) * (max - 1));

    int count = 2;

    for (int i = 1; i < max; i++) {
        int index = 0;
        for (int c = 0; c < count; c++) {
            for (int n = 0; n < max; n++) {
                const double v = n == i ? 1 : result[index];
                result[count * max + c * max + n] = v;

                index++;
            }
        }

        count *= 2;
    }
}

void generate_classify_sum_outputs(double* result, const double* inputs, const int max) {
    const int count = (int)pow(2, max);

    for (int i = 0; i < count; i++) {
        int sum = 0;
        for (int j = 0; j < max; j++) {
            sum += (int)inputs[i * max + j];
        }

        for (int j = 0; j < max + 1; j++) {
            result[i * (max + 1) + j] = sum == j ? 1 : 0;
        }
    }
}

void generate_binary_sum_outputs(double* result, const double* inputs, const int max) {
    const int count = (int)pow(2, max);
    const int representation = (int)ceil(log2(max));

    for (int i = 0; i < count; i++) {
        int sum = 0;
        for (int j = 0; j < max; j++) {
            sum += (int)inputs[i * max + j];
        }


        for (int j = 0; j < representation; j++) {
            result[i * representation + j] = (sum >> j & 1) == 1 ? 1 : 0;
        }
    }
}
