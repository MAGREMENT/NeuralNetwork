//
// Created by zacha on 21-10-25.
//

#include "optimizer.h"

#include <stddef.h>

inline void* cnstr_empty_state(const optimizer* opt, const neural_network* network) {
    return NULL;
}

inline void free_empty_state(void* state, const neural_network* network) {}