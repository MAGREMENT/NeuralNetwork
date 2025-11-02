//
// Created by zacha on 01-10-25.
//

#ifndef MINI_BATCH_ITERATOR_H
#define MINI_BATCH_ITERATOR_H

#include "../iterator.h"

range_iterator* cnstr_mini_batch_iterator(int size, int iterations, int batchSize);

#endif //MINI_BATCH_ITERATOR_H
