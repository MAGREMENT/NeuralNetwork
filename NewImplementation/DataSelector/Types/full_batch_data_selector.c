//
// Created by zacha on 28-10-25.
//

#include "../Types/full_batch_data_selector.h"

#include <stdlib.h>

#include "../../Iterators/Types/full_batch_iterator.h"

static range_iterator* cnstr_iterator(const data_selector* sel, const int dataSize, const int iterations) {
    return cnstr_full_batch_iterator(dataSize, iterations);
}

data_selector_vtable vtable = {cnstr_iterator, def_free_ds};

inline data_selector* cnstr_full_batch_data_selector() {
    data_selector* sel = malloc(sizeof(data_selector));

    sel->params = NULL;
    sel->vtable = vtable;

    return sel;
}