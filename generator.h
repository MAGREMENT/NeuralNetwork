#ifndef GENERATOR_H
#define GENERATOR_H

struct test_data {
    int count;
    struct input_data* inputs;
    struct input_data* expected;
};

void free_test_data(struct test_data* data);
struct test_data* positive_generate_for_2(double spacing, int count, int max, int(*cut)(double, double));

#endif //GENERATOR_H
