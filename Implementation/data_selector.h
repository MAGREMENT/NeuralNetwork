//
// Created by zacha on 01-10-25.
//

#ifndef DATA_SELECTOR_H
#define DATA_SELECTOR_H
#include "Iterators/iterator.h"

typedef struct data_selector data_selector;

struct data_selector {
    void* params;
    range_iterator* (*constr_iterator)(data_selector* sel, int dataSize, int iterations);
    void (*free)(data_selector* selector);
};

data_selector* create_full_batch_selector();

#endif //DATA_SELECTOR_H
