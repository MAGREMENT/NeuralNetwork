//
// Created by zacha on 16-11-25.
//

#include "bitset.h"

#include <stdlib.h>

inline int* alloc_bitset(const int capacity) {
    return malloc(sizeof(int) * (capacity / sizeof(int) + 1));
}

inline int is_set(const int* bitset, const int index) {
    const int i = bitset[index / sizeof(int)];
    return (i >> (index % sizeof(int)) & 1) == 1;
}

inline void set(int* bitset, const int index) {
    bitset[index / sizeof(int)] |= 1 << (index % sizeof(int));
}

inline void unset(int* bitset, const int index) {
    bitset[index / sizeof(int)] &= ~(1 << (index % sizeof(int)));
}