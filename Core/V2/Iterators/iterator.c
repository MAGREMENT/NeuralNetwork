//
// Created by zacha on 22-10-25.
//

#include "iterator.h"

#include <stdlib.h>

inline void default_free_iterator(range_iterator *iterator) {
    free(iterator->state);
    free(iterator);
}
