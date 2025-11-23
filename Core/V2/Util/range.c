//
// Created by zacha on 17-11-25.
//

#include "range.h"

inline range_split_iterator range_split(const int start, const int total, const int count) {
    return (range_split_iterator) {total / count, start,  total % count};
}

inline range range_split_next(range_split_iterator* iterator) {
    const int from = iterator->curr;

    iterator->curr += iterator->per;
    if (iterator->add > 0) {
        iterator->curr++;
        iterator->add--;
    }

    return (range) {from, iterator->curr};
}

inline range to_range(const iteration_range ir) {
    return (range) {ir.from, ir.to};
}

inline iteration_range to_iteration_range(const range range, const int iteration) {
    return (iteration_range) {range.from, range.to, iteration};
}