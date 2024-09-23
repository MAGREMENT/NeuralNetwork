#include "utils.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

inline double random(const double from, const double to) {
    return (double)rand() / RAND_MAX * (to - from) + from;
}

inline int max_index(double values[], const int count) {
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
