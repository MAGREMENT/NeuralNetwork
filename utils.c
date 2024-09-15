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

inline int equals(const double left, const double right, const double margin) {
    return fabs(left - right) < margin;
}

int default_equals(const double left, const double right) {
    return equals(left, right, 0.00001);
}

struct batch create_batch(int current, int size, int max) {
    struct batch result;
    if(size == max) {
        result.to = max;
        result.then = current;
        return result;
    }

    int to = current + size;
    if(to >= max) {
        result.to = max;
        result.then = to % max;
    }
    else {
        result.to = to;
        result.then = 0;
    }

    return result;
}
