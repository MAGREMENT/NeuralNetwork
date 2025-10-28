//
// Created by zacha on 28-10-25.
//

#ifndef NEWIMPLEMENTATION_DATA_SELECTOR_H
#define NEWIMPLEMENTATION_DATA_SELECTOR_H

#include "../Iterators/iterator.h"

typedef struct data_selector data_selector;

typedef struct data_selector_vtable {
    range_iterator* (*cnstr_iterator)(const data_selector* sel, int dataSize, int iterations);
    void (*free)(data_selector* selector);
} data_selector_vtable;

struct data_selector {
    void* params;
    data_selector_vtable vtable;
};

void def_free_ds(data_selector* selector);

#endif //NEWIMPLEMENTATION_DATA_SELECTOR_H