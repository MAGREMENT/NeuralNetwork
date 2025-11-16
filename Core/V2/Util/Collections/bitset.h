//
// Created by zacha on 16-11-25.
//

#ifndef NEWIMPLEMENTATION_BITSET_H
#define NEWIMPLEMENTATION_BITSET_H

extern int* alloc_bitset(int capacity);
extern int is_set(const int* bitset, int index);
extern void set(int* bitset, int index);
extern void unset(int* bitset, int index);

#endif //NEWIMPLEMENTATION_BITSET_H