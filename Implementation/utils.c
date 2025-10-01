#include "utils.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>


inline void init_random() {
    srand(time(NULL));
}

inline double rand_d(const double min, const double max) {
    return (double)rand() / (double)RAND_MAX * (max - min) + min;

}

inline int rand_i(const int max) {
    return rand() % max;
}

inline int rand_std_nrml_distribution() {
    const double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);  // avoid log(0)
    const double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
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
