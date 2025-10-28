//
// Created by zacha on 28-10-25.
//

#include "mini_batch_data_selector.h"

#include <stdlib.h>

#include "../../Iterators/Types/mini-batch_iterator.h"

static range_iterator* cnstr_iterator(const data_selector* sel, const int dataSize, const int iterations) {
    return constr_mini_batch_iterator(dataSize, iterations, *(int*)sel->params);
}

static void free_ds(data_selector* selector) {
    free(selector->params);
    free(selector);
}

data_selector_vtable mbds_vtable = {cnstr_iterator, free_ds};

inline data_selector* cnstr_mini_batch_data_selector(const int size) {
    data_selector* sel = malloc(sizeof(data_selector));
    int* s = malloc(sizeof(int));
    *s = size;

    sel->params = s;
    sel->vtable = mbds_vtable;

    return sel;
}