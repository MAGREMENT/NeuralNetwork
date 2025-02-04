#ifndef GENERATOR_H
#define GENERATOR_H

#include "neural_network.h"

/**
 * Generate test data in a 2d space
 * @param spacing
 * @param count
 * @param max
 * @param cut
 * @return
 */
test_data* positive_generate_for_2D(double spacing, int count, int max, int(*cut)(double, double));

#endif //GENERATOR_H
