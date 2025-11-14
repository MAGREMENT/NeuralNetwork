//
// Created by zacha on 01-10-25.
//

#include "mini-batch_iterator.h"
#include "../iterator-util.h"

#include <stdlib.h>

typedef struct mini_batch_iterator_state {
    int max_iterations;
    int size;
    int batch_size;
} mini_batch_iterator_state;

static int next(range_iterator* iterator) {
    mini_batch_iterator_state* state = iterator->state;
    if (state->max_iterations <= 0) return false;

    if (iterator->current.to >= state->size) {
        iterator->current.iteration++;
        if (iterator->current.iteration > state->max_iterations) return false;

        iterator->current.from = 0;
        iterator->current.to = state->batch_size;
    } else {
        iterator->current.from = iterator->current.to;
        iterator->current.to += state->batch_size;
    }

    if (iterator->current.to > state->size) iterator->current.to = state->size;

    return iterator->current.iteration <= state->max_iterations;
}

static void reset(range_iterator* iterator) {
    mini_batch_iterator_state* state = iterator->state;

    iterator->current.iteration = 0;
    iterator->current.from = 0;
    iterator->current.to = state->size;
}

inline range_iterator* constr_mini_batch_iterator(int size, int iterations, int batchSize) {
    range_iterator* it = malloc(sizeof(range_iterator));
    mini_batch_iterator_state* state = malloc(sizeof(mini_batch_iterator_state));

    state->max_iterations = iterations;
    state->size = size;
    state->batch_size = batchSize;
    it->state = state;

    it->current.iteration = 0;
    it->current.from = 0;
    it->current.to = size;

    it->next = next;
    it->free = default_free_iterator;
    it->reset = reset;

    return it;
}