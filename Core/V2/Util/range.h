//
// Created by zacha on 17-11-25.
//

#ifndef NEWIMPLEMENTATION_RANGE_H
#define NEWIMPLEMENTATION_RANGE_H

typedef struct range {
    int from;
    int to;
} range;

typedef struct range_split_iterator {
    int per;
    int curr;
    int add;
} range_split_iterator;

extern range_split_iterator range_split(int total, int count);
extern range range_split_next(range_split_iterator* iterator);

#endif //NEWIMPLEMENTATION_RANGE_H