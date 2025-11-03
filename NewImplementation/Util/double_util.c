//
// Created by zacha on 02-11-25.
//

#include "double_util.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

inline int d_max_ind(const double* values, const int count) {
    double max = DBL_MIN;
    int index = -1;
    for(int i = 0; i < count; i++) {
        if(values[i] > max) {
            index = i;
            max = values[i];
        }
    }

    return index;
}

inline int deq(const double left, const double right, const double margin) {
    return fabs(left - right) < margin;
}

int def_deq(const double left, const double right) {
    return deq(left, right, 0.00001);
}

char* alloc_dseq_to_str(const double* values, const int count) {
    int size = 0;
    for (int i = 0; i < count; i++) {
        size += snprintf(NULL, 0, "%.2f", values[i]);
        if (i != 0) size += 2;
    }

    char* result = malloc(sizeof(char) * (size + 1));
    char* current = result;
    for (int i = 0; i < count; i++) {
        if (i != 0) {
            current[0] = ',';
            current[1] = ' ';
            current += 2;
        }

        int n = snprintf(current, size, "%.2f", values[i]);
        current += n;
    }

    current[0] = '\0';
    return result;
}