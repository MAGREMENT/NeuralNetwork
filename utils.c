#include "utils.h"

#include <float.h>
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
