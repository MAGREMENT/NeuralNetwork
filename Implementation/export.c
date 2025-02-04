#include "export.h"
#include "functions.h"
#include "utils.h"

inline neural_network* Initialize() {
    const int numbers[] = {2, 3, 2};
    return alloc_network(3, numbers);
}

inline int ExportTest(const int a) {
    return a * 2;
}