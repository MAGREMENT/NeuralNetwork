//
// Created by zacha on 01-10-25.
//

#include "full_batch_iterator.h"

#include <stdlib.h>
#include "../iterator-util.h"

static int next(range_iterator* iterator) {
     const int max = *(int*)iterator->state;

    iterator->current.iteration++;
    return iterator->current.iteration <= max;
}

static void reset(range_iterator* iterator) {
    iterator->current.iteration = 0;
}

inline range_iterator* constr_full_batch_iterator(int size, int iterations) {
    range_iterator* it = malloc(sizeof(range_iterator));
    int* state = malloc(sizeof(int));

    *state = iterations;
    it->state = state;

    it->current.iteration = 0;
    it->current.from = 0;
    it->current.to = size;

    it->next = next;
    it->free = default_free_iterator;
    it->reset = reset;

    return it;
}
