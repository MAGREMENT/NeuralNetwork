#include "utils.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


inline void init_random() {
    srand(time(NULL));
}

inline double random(const double min, const double max) {
    return (double)rand() / (double)RAND_MAX * (max - min) + min;

}

inline int rand_i(const int max) {
    return rand() % max;
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

void list_remove(int* arr, int count, int index) {
    index++;
    for (; index < count; index++) {
        arr[index - 1] = arr[index];
    }
}
