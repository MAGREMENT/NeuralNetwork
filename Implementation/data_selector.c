//
// Created by zacha on 01-10-25.
//

#include "data_selector.h"

#include <stdlib.h>

#include "Iterators/Implementations/full_batch_iterator.h"

static range_iterator* constr_fb_iterator(data_selector* sel, int dataSize, int iterations) {
    return constr_full_batch_iterator(dataSize, iterations);
}

data_selector* create_full_batch_selector() {
    data_selector* sel = malloc(sizeof(data_selector));

    sel->free = free;
    sel->constr_iterator = constr_fb_iterator;

    return sel;
}
