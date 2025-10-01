//
// Created by zacha on 01-10-25.
//

#include "full_batch_iterator.h"

#include <stdlib.h>
#include "../iterator-util.h"

typedef struct full_batch_iterator_state {
    int started;
    int max_iterations;
} full_batch_iterator_state;

static int next(range_iterator* iterator) {
    full_batch_iterator_state* state = iterator->state;
    if (!state->started) {
        state->started = true;
        return true;
    }

    iterator->current.iteration++;
    return iterator->current.iteration < state->max_iterations;
}

static void reset(range_iterator* iterator) {
    full_batch_iterator_state* state = iterator->state;
    iterator->current.iteration = 0;
    state->started = false;
}

inline range_iterator* constr_full_batch_iterator(int size, int iterations) {
    range_iterator* it = malloc(sizeof(range_iterator));
    full_batch_iterator_state* state = malloc(sizeof(full_batch_iterator_state));

    state->started = false;
    state->max_iterations = iterations;
    it->state = state;

    it->current.iteration = 0;
    it->current.from = 0;
    it->current.to = size;

    it->next = next;
    it->free = default_free_iterator;
    it->reset = reset;

    return it;
}
