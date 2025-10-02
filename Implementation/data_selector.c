//
// Created by zacha on 01-10-25.
//

#include "data_selector.h"

#include <stdlib.h>

#include "Iterators/Implementations/full_batch_iterator.h"
#include "Iterators/Implementations/mini-batch_iterator.h"

static range_iterator* constr_fb_iterator(data_selector* sel, int dataSize, int iterations) {
    return constr_full_batch_iterator(dataSize, iterations);
}

inline data_selector* create_full_batch_selector() {
    data_selector* sel = malloc(sizeof(data_selector));

    sel->free = free;
    sel->constr_iterator = constr_fb_iterator;

    return sel;
}

static void free_base(data_selector* sel) {
    free(sel->params);
    free(sel);
}

static range_iterator* constr_mb_iterator(data_selector* sel, int dataSize, int iterations) {
    return constr_mini_batch_iterator(dataSize, iterations, *(int*)sel->params);
}

inline data_selector* create_mini_batch_selector(int batchSize) {
    data_selector* sel = malloc(sizeof(data_selector));
    int* bs = malloc(sizeof(int));
    *bs = batchSize;

    sel->params = bs;
    sel->free = free_base;
    sel->constr_iterator = constr_mb_iterator;

    return sel;
}
