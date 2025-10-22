#ifndef ITERATOR_H
#define ITERATOR_H

typedef struct range {
    int iteration;
    int from;
    int to;
} range;

typedef struct range_iterator range_iterator;

struct range_iterator {
    void* state;
    int (*next)(range_iterator* self);
    void (*free)(range_iterator* self);
    void (*reset)(range_iterator* self);
    range current;
};

void default_free_iterator(range_iterator *iterator);

#endif //ITERATOR_H
