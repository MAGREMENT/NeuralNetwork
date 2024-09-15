#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "neural_network.h"
#include "generator.h"
#include "functions.h"
#include "utils.h"
#include "repository.h"

void unit_tests();
void program();

int main() {
    //program();
    unit_tests();

    return EXIT_SUCCESS;
}

struct neural_network* example_network(int activation) {
    int numbers[] = {2, 3, 2};
    struct neural_network* network = alloc_network(3, numbers);
    struct params params;
    params.learningRate = 1;
    params.activationType = activation;
    params.costType = MEAN_SQUARED;
    apply_params(network, params);

    return network;
}

struct neural_network* example_network_with_data(int activation) {
    struct neural_network* network = example_network(activation);

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
    struct neural_network* network = example_network(SIGMOID);

    randomize(network, -1, 1);
    struct test_data *test = positive_generate_for_2(0.5, 20, 2, parabole_cut_10);

    test_network(network, test);
    int current = 0;
    for(int iteration = 0; iteration < 1000; iteration++) {
        const struct batch b = create_batch(current, test->count, test->count);
        learn(network, test->inputs, test->expected, current, b.to, b.then);
        current = b.then == 0 ? b.to : b.then;

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
    struct neural_network* network = example_network_with_data(DEFAULT);

    struct input_data* input = alloc_input_data(2);
    input->values[0] = 2;
    input->values[1] = 1;
    struct backpropagation_data* data = traverse(network, input);

    struct backpropagation_data* expected = alloc_back_data(network);
    expected[0].weightedInputs[0] = 0.5;
    expected[0].weightedInputs[1] = 3;
    expected[0].weightedInputs[2] = 3.5;
    expected[1].weightedInputs[0] = 2;
    expected[1].weightedInputs[1] = 6.5;

    for(int l = 0; l < data->count; l++) {
        for(int o = 0; o < data[l].count; o++) {
            const double w = data[l].weightedInputs[o];
            const double e = expected[l].weightedInputs[o];
            if(!default_equals(w, e)) {
                printf("invalid input weight at layer %d and index %d, expected %.4f, got %.4f\n",l, o, e, w);
                return;
            }
        }
    }

    free_back_data(data, network->count);
    free_back_data(expected, network->count);
    free_network(network);
    printf("traverse test OK!\n");
}

void learn_test() {
    struct neural_network* network = example_network_with_data(SIGMOID);
    struct test_data *test = positive_generate_for_2(0.5, 20, 2, diagonal_cut);

    double cost = multi_cost(network, test->inputs, test->expected, test->count);
    for(int i = 0; i < 20; i++) {
        learn(network, test->inputs, test-> expected, 0, test->count, 0);
        double nCost = multi_cost(network, test->inputs, test->expected, test->count);

        if(nCost > cost + 1) {
            printf("new cost (%.5f) bigger by more than 1 than previous cost (%.5f) on iteration %d\n", nCost, cost, i);
            return;
        }
    }

    free_network(network);
    free_test_data(test);
    printf("learn test OK!\n");
}

void gradients_test() {
    struct neural_network* network = example_network_with_data(SIGMOID);
    struct input_data* input = alloc_input_data(2);
    input->values[0] = 2;
    input->values[1] = 1;

    struct input_data* expected = alloc_input_data(2);
    expected->values[0] = 1;
    expected->values[1] = 0;

    struct gradients* gradients = alloc_gradients(network, 0);
    update_gradients(network, gradients, *input, *expected);

    /* TODO
    i = 2  e = 1
      = 1    = 0

    z1 = 0.5  a1 = 0.62245  z2 = -0.1047  a2 = 0.47384
       = 3       = 0.95257     = 1.02198     = 0.73535
       = 3.5     = 0.97068

    cd = -1.05232  ad = 0.24931  nv = -0.26235
       = 1.4707       = 0.19461     = 0.28621

    wd = -0.10731
       = 0.29274
       = 0.15503
    */

    free_network(network);
    free_input_data(input);
    free_input_data(expected);
    printf("gradients test OK!\n");
}

void repository_test() {
    constexpr char filename[] = "neural_network_repository_test.nn";
    FILE* fptr = fopen(filename, "w");
    fclose(fptr);

    struct neural_network* network = example_network_with_data(SIGMOID);
    struct params params;
    params.learningRate = 1;
    params.activationType = SIGMOID;
    params.costType = MEAN_SQUARED;
    apply_params(network, params);

    save(network, &params, filename);
    struct neural_network* download = initialize(filename);

    if(network->count != download->count) {
        printf("Different network count");
        return;
    }

    if(network->learningRate != download->learningRate) {
        printf("Different learning rate");
        return;
    }

    if(network->cost != download->cost) {
        printf("Different cost function");
        return;
    }

    for(int i = 0; i < network->count; i++) {
        if(network->layers[i].in_count != download->layers[i].in_count) {
            printf("Different in count for layer %d", i);
            return;
        }

        if(network->layers[i].out_count != download->layers[i].out_count) {
            printf("Different out count for layer %d", i);
            return;
        }

        if(network->layers[i].activation != download->layers[i].activation) {
            printf("Different activation for layer %d", i);
            return;
        }

        const int in = network->layers[i].in_count;
        const int out = network->layers[i].out_count;

        for(int o = 0; o < out; o++) {
            if(!default_equals(network->layers[i].biases[o], download->layers[i].biases[o])) {
                printf("Different bias for layer %d and output %d", i, o);
                return;
            }

            for(int j = 0; j < in; j++) {
                if(!default_equals(network->layers[i].weights[j * out + o], download->layers[i].weights[j * out + o])) {
                    printf("Different weights for layer %d, output %d and input %d", i, o, j);
                    return;
                }
            }
        }
    }

    free_network(network);
    free_network(download);
    remove(filename);
    printf("repository test OK!\n");
}

void unit_tests() {
    generate_test(0);
    learn_test();
    traverse_test();
    gradients_test();
    repository_test();
}

