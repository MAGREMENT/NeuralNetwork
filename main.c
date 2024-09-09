#include <stdio.h>
#include <stdlib.h>

#include "neural_network.h"
#include "generator.h"
#include "functions.h"

void unit_tests();
void program();

int main() {
    program();
    //unit_tests();

    return EXIT_SUCCESS;
}

void program() {
    int numbers[] = {2, 3, 2};
    struct neural_network* network = alloc_network(3, numbers);
    struct params params;
    params.learningRate = 0.8;
    params.activationType = SIGMOID;
    params.costType = MEAN_SQUARED;
    apply_params(network, params);

    randomize(network, -10, 10);
    const struct test_data *test = positive_generate_for_2(0.5, 20, 2, diagonal_cut);

    for(int iteration = 0; iteration < 20; iteration++) {
        learn(network, test->inputs, test->expected, test->count);

        int valid = 0;
        for(int i = 0; i < test->count; i++) {
            struct input_data* output = predict(network, &test->inputs[i]);
            if(is_valid(output, &test->expected[i])) valid++;
            free_input_data(output);
        }

        printf("accuracy : %f / 100.0\n", (double)valid / test->count * 100);
    }
}

void generate_test() {
    const struct test_data *test = positive_generate_for_2(0.5, 20, 2, diagonal_cut);
    if(test->count != 400) {
        printf("invalid count");
        return;
    }

    for(int i = 0; i < test->count; i++) {
        if(test->inputs[i].count != 2) {
            printf("invalid input count at %d", i);
            return;
        }

        if(test->inputs[i].values[0] < 0 || test->inputs[i].values[1] < 0) {
            printf("negative input at %d", i);
            return;
        }

        if(test->expected[i].count != 2) {
            printf("invalid expected count at %d", i);
            return;
        }

        if(test->expected[i].values[0] < 0 || test->expected[i].values[1] < 0) {
            printf("negative expected at %d", i);
            return;
        }
    }

    printf("generate test OK!");
}

void unit_tests() {
    generate_test();
}

