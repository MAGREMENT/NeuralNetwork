#include <stdio.h>
#include <stdlib.h>

#include "neural_network.h"
#include "generator.h"
#include "functions.h"
#include "utils.h"

void unit_tests();
void program();

int main() {
    program();
    //unit_tests();

    return EXIT_SUCCESS;
}

struct neural_network* example_network() {
    int numbers[] = {2, 3, 2};
    struct neural_network* network = alloc_network(3, numbers);
    struct params params;
    params.learningRate = 1;
    params.activationType = SIGMOID;
    params.costType = MEAN_SQUARED;
    apply_params(network, params);

    return network;
}

struct neural_network* example_network_with_data() {
    struct neural_network* network = example_network();

    double w1[] = {0.5, 1, 1.5, 0.5, 1, 1.5};
    double w2[] = {1.5, 1, 0.5, 1.5, 0.5, 1};
    double b1[] = {-1, 0, -1};
    double b2[] = {-2, -2};

    set_layer(network->layers[0], w1, b1);
    set_layer(network->layers[1], w2, b2);

    return network;
}

void print_network(struct neural_network* network) {
    for(int i = 0; i < network->count; i++) {
        printf("weights %d : ", i);

        for(int o = 0; o < network->layers[i].out_count; o++) {
            for(int j = 0; j < network->layers[i].in_count; j++) {
                printf("%.2f ", network->layers[i].weights[j * network->layers[i].out_count + o]);
            }
        }

        printf("\nbiases %d : ", i);
        for(int o = 0; o < network->layers[i].out_count; o++) {
            printf("%.2f ", network->layers[i].biases[o]);
        }

        printf("\n");
    }
}

void test_network(struct neural_network* network, struct test_data *test) {
    int valid = 0;
    for(int i = 0; i < test->count; i++) {
        struct input_data* output = predict(network, &test->inputs[i]);
        if(is_valid(output, &test->expected[i])) valid++;
        free_input_data(output);
    }

    print_network(network);
    printf("   cost : %.5f\n", multi_cost(network, test->inputs, test->expected, test->count));
    printf("   accuracy : %.2f / 100.0\n", (double)valid / test->count * 100);
}

void program() {
    struct neural_network* network = example_network();

    randomize(network, -1, 1);
    struct test_data *test = positive_generate_for_2(0.5, 20, 2, parabole_cut_10);

    test_network(network, test);
    for(int iteration = 0; iteration < 100; iteration++) {
        learn(network, test->inputs, test->expected, test->count);
        test_network(network, test);
    }

    free_network(network);
}

void generate_test(int verbose) {
    struct test_data *test = positive_generate_for_2(0.5, 20, 2, diagonal_cut);
    if(test->count != 400) {
        printf("invalid count\n");
        return;
    }

    for(int i = 0; i < test->count; i++) {
        if(test->inputs[i].count != 2) {
            printf("invalid input count at %d\n", i);
            return;
        }

        if(test->inputs[i].values[0] < 0 || test->inputs[i].values[1] < 0) {
            printf("negative input at %d\n", i);
            return;
        }

        if(test->expected[i].count != 2) {
            printf("invalid expected count at %d\n", i);
            return;
        }

        if(test->expected[i].values[0] < 0 || test->expected[i].values[1] < 0) {
            printf("negative expected at %d\n", i);
            return;
        }
    }

    if(verbose) {
        for(int i = 0; i < 41; i++) {
            printf("%.2f %.2f -> %.2f %.2f\n", test->inputs[i].values[0], test->inputs[i].values[1],
                test->expected[i].values[0], test->expected[i].values[1]);
        }
    }

    free_test_data(test);
    printf("generate test OK!\n");
}

void traverse_test() {
    struct neural_network* network = example_network_with_data();
    network->activation = default_activation;

    struct input_data* input = alloc_input_data(2);
    input->values[0] = 2;
    input->values[1] = 1;
    struct backpropagation_data* data = traverse(network, input);

    struct backpropagation_data* expected = alloc_back_data(network);
    expected->layers[0].weightedInputs[0] = 0.75;
    expected->layers[0].weightedInputs[1] = 3;
    expected->layers[0].weightedInputs[2] = 3.5;
    expected->layers[1].weightedInputs[0] = 2.375;
    expected->layers[1].weightedInputs[1] = 6.75;

    for(int l = 0; l < data->count; l++) {
        for(int o = 0; o < data->layers[l].count; o++) {
            const double w = data->layers[l].weightedInputs[o];
            const double e = expected->layers[l].weightedInputs[o];
            if(!default_equals(w, e)) {
                printf("invalid input weight at layer %d and index %d, expected %.4f, got %.4f\n",l, o, e, w);
                return;
            }
        }
    }

    free_network(network);
    free_back_data(data);
    free_back_data(expected);
    printf("traverse test OK!\n");
}

void learn_test() {
    struct neural_network* network = example_network_with_data();
    struct test_data *test = positive_generate_for_2(0.5, 20, 2, diagonal_cut);

    double cost = multi_cost(network, test->inputs, test->expected, test->count);
    for(int i = 0; i < 20; i++) {
        learn(network, test->inputs, test-> expected, test->count);
        double nCost = multi_cost(network, test->inputs, test->expected, test->count);

        if(nCost > cost + 10) {
            printf("new cost (%.5f) bigger by more than 10 than previous cost (%.5f) on iteration %d\n", nCost, cost, i);
            return;
        }
    }

    free_network(network);
    free_test_data(test);
    printf("learn test OK!\n");
}

void unit_tests() {
    generate_test(0);
    learn_test();
    traverse_test();
}

