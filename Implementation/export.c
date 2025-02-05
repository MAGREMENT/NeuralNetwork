#include "export.h"
#include "functions.h"
#include "utils.h"

inline neural_network* Initialize(int count, int numbers[]) {
    return alloc_network(count, numbers);
}

inline int GetCount(neural_network* ptr) {
    return ptr->count;
}