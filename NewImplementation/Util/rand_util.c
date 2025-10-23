//
// Created by zacha on 23-10-25.
//

#include "rand_util.h"

#include <float.h>
#include <stdlib.h>
#include <tgmath.h>
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

inline double rand_d_std_nrml_distr() {
    const double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);  // avoid log(0)
    const double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}