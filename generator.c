#include "generator.h"
#include "neural_network.h"

#include <stdlib.h>

inline test_data* positive_generate_for_2(const double spacing, const int count, const int max, int(*cut)(double, double)) {
    test_data* result = alloc_test_data(count * count);

    for(int i = 0; i < count; i++) {
        for(int j = 0; j < count; j++) {
            const double x = i * spacing;
            const double y = j * spacing;
            const int index = i * count + j;

            result->inputs[index].count = 2;
            result->inputs[index].values = malloc(2 * sizeof(double));
            result->inputs[index].values[0] = x;
            result->inputs[index].values[1] = y;

            result->expected[index].count = max;
            result->expected[index].values = malloc(max * sizeof(double));
            const int e = cut(x, y);
            for(int n = 0; n < max; n++) {
                result->expected[index].values[n] = n == e;
            }
        }
    }

    return result;
}
