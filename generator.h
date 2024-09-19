#ifndef GENERATOR_H
#define GENERATOR_H

typedef struct test_data {
    int count;
    struct input_data* inputs;
    struct input_data* expected;
} test_data;

void free_test_data(test_data* data);
test_data* positive_generate_for_2(double spacing, int count, int max, int(*cut)(double, double));

#endif //GENERATOR_H
