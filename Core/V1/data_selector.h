//
// Created by zacha on 01-10-25.
//

#ifndef DATA_SELECTOR_H
#define DATA_SELECTOR_H
#include "list.h"
#include "Iterators/iterator.h"

typedef struct data_selector data_selector;

struct data_selector {
    void* params;
    range_iterator* (*constr_iterator)(data_selector* sel, int dataSize, int iterations);
    void (*free)(data_selector* selector);
    s_arr* (*alloc_to_hyper)(data_selector* ds, int startIndentation); //TODO finish implementation
};

data_selector* create_full_batch_selector();
data_selector* create_mini_batch_selector(int batchSize);

#endif //DATA_SELECTOR_H
