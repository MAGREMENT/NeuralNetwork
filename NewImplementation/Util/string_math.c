//
// Created by zacha on 22-10-25.
//

#include "string_math.h"

#include <tgmath.h>

inline double sigmoid(const double input) {
    return 1 / (1 + exp(-input));
}
