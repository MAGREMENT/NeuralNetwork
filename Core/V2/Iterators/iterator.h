#ifndef ITERATOR_H
#define ITERATOR_H

#include "../Util/range.h"

typedef struct range_iterator range_iterator;

struct range_iterator {
    void* state;
    int (*next)(range_iterator* self);
    void (*free)(range_iterator* self);
    void (*reset)(range_iterator* self);
    iteration_range current;
};

extern void default_free_iterator(range_iterator *iterator);

#endif //ITERATOR_H
