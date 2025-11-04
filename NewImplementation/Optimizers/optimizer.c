//
// Created by zacha on 21-10-25.
//

#include "optimizer.h"

#include <stddef.h>
#include <stdlib.h>

inline void* cnstr_empty_state(const optimizer* opt, const neural_network* network) {
    return NULL;
}

inline void free_empty_state(void* state, const neural_network* network) {}

inline void* cnstr_gradient_buffers_state(const optimizer* opt, const neural_network* network) {
    return alloc_gradient_buffers(network, true);
}

inline void free_gradient_buffers_state(void* state, const neural_network* network) {
    free_buffers(network, state);
}

inline void* cnstr_double_gradient_buffers_state(const optimizer* opt, const neural_network* network) {
    double*** /*xD*/ s = malloc(sizeof(double**) * 2);
    s[0] = alloc_gradient_buffers(network, true);
    s[1] = alloc_gradient_buffers(network, true);
    return s;
}

inline void free_double_gradient_buffers_state(void* state, const neural_network* network) {
    double*** s = state;
    free_buffers(network, s[0]);
    free_buffers(network, s[1]);
    free(s);
}

inline void free_empty_opt(optimizer* opt) {
    free(opt);
}

inline void free_base_opt(optimizer* opt) {
    free(opt->params);
    free(opt);
}