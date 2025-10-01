//
// Created by zacha on 01-10-25.
//

#include "iterator-util.h"

#include <stdlib.h>

void default_free_iterator(range_iterator* iterator) {
    free(iterator->state);
    free(iterator);
}
