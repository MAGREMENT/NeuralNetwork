#include "generator.h"
#include "old_nn.h"

#include <stdlib.h>

inline test_data* positive_generate_for_2D(const double spacing, const int count, const int max, int(*cut)(double, double)) {
    test_data* result = alloc_test_data(count * count, 2, max);

    for(int i = 0; i < count; i++) {
        for(int j = 0; j < count; j++) {
            const double x = i * spacing;
            const double y = j * spacing;
            const int index = i * count + j;

            result->inputs[index].count = 2;
            result->inputs[index].values[0] = x;
            result->inputs[index].values[1] = y;

            const int e = cut(x, y);
            for(int n = 0; n < max; n++) {
                result->expected[index].values[n] = n == e;
            }
        }
    }

    return result;
}
