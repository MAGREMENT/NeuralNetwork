#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "neural_network.h"
#include "big-array.c"
#include "functions.h"
#include "generator.h"
#include "repository.h"
#include "utils.h"

void unit_tests();
void cut_2D_test();
void yolo();

int main() {
    //cut_2D_test();
    //unit_tests();
    yolo();

    return EXIT_SUCCESS;
}

void yolo() {
    clock_t start = clock();

    const int numbers[] = {7, 4, 8};
    neural_network* network = alloc_network(3, numbers);
    apply_default_hyper_params(network);
    set_activation_type(network, SIGMOID, SOFTMAX);
    set_cost_type(network, BINARY_CROSS_ENTROPY);
    set_scheduler(network, constr_constant_scheduler());
    set_optimizer(network, create_nesterov_optimizer(0.9));
    network->learningRate = 1;
    network->threadCount = 1;

    /*list* l = alloc_get_hyper_params(network);
    save_yaml(l->data, l->count, "test.yaml");
    free_list(l);

    return EXIT_SUCCESS;*/

    //test_data* data = positive_generate_for_2D(1, 100, 2, parable_10_cut);
    test_data* data = alloc_transfer_flattened_data(big_arr3, 7, big_arr4, 8, 128);

    //initialize(network);
    set_all_weights_and_biases(network, 1, 1);

    gradient_diagnostic* diag = alloc_run_gradient_diagnostic(network, data, -5, 5);
    print_diagnostic(network, diag);

    //set_all_weights_and_biases(network, 1, 1);

    iterative_learn(network, data, NULL, 1000);

    /*learning_state* state = alloc_state(network);
    for (int i = 0; i < 10; i++) {
        iterative_learn(network, data, state, 100);
    }

    free_state(network, state);*/

    printf("%f\n", avg_cost(network, data));

    /*double accuracy = 0;
    for (int i = 0; i < data->count; i++) {
        input_data result;
        result.count = 3;
        double val[3];
        result.values = val;

        predict(network, &data->inputs[i], &result);
        int ok = true;
        for (int j = 0; j < 3; j++) {
            if (fabs(round(val[j]) - data->expected[i].values[j]) > 0.1) ok = false;
        }

        if (ok) accuracy++;
    }

    printf("%f\n", accuracy / data->count * 100);*/

    free_gradient_diagnostic(diag);
    diag = alloc_run_gradient_diagnostic(network, data, -5, 5);
    print_diagnostic(network, diag);

    free_gradient_diagnostic(diag);
    free_network(network);
    free_transferred_flattened_data(data);

    clock_t end = clock();

    printf("Time : %fs", (double)(end - start) / CLOCKS_PER_SEC);
}

neural_network* alloc_example_network(int activation) {
    int numbers[] = {2, 3, 2};
    neural_network* network = alloc_network(3, numbers);
    apply_default_hyper_params(network);
    initialize(network);

    return network;
}

neural_network* alloc_example_network_with_data(int activation) {
    neural_network* network = alloc_example_network(activation);
    set_activation_type(network, activation, activation);

    double w1[] = {0.5, 1, 1.5, 0.5, 1, 1.5};
    double w2[] = {1.5, 1, 0.5, 1.5, 0.5, 1};
    double b1[] = {-1, 0, -1};
    double b2[] = {-2, -2};

    set_layer(network->layers[0], w1, b1);
    set_layer(network->layers[1], w2, b2);

    return network;
}

void print_network(neural_network* network) {
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

void test_and_print_network(neural_network* network, test_data* data, const int i) {
    const test_result result = test_network(network, data);

    print_network(network);
    printf("   cost : %.5f\n", result.cost);
    printf("   accuracy : %.2f / 100.0\n", result.accuracy);
}

void cut_2D_test() {
    neural_network* network = alloc_example_network(SIGMOID);

    initialize(network);
    test_data *test = positive_generate_for_2D(0.5, 20, 2, sinus_cut);

    test_and_print_network(network, test, -1);
    iterative_learn(network, test, NULL, 1000);

    free_network(network);
}

void generate_test(const int verbose) {
    test_data *test = positive_generate_for_2D(0.5, 20, 2, diagonal_cut);
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
    neural_network* network = alloc_example_network_with_data(DEFAULT);

    input_data* input = alloc_input_data(2);
    input->values[0] = 2;
    input->values[1] = 1;
    backpropagation_data* data = alloc_traverse(network, input);

    backpropagation_data* expected = alloc_back_data(network);
    expected[0].weightedInputs[0] = 0.5;
    expected[0].weightedInputs[1] = 3;
    expected[0].weightedInputs[2] = 3.5;
    expected[1].weightedInputs[0] = 2;
    expected[1].weightedInputs[1] = 6.5;

    for(int l = 0; l < network->count; l++) {
        for(int o = 0; o < data[l].count; o++) {
            const double w = data[l].weightedInputs[o];
            const double e = expected[l].weightedInputs[o];
            if(!def_deq(w, e)) {
                printf("invalid input weight at layer %d and index %d, expected %.4f, got %.4f\n",l, o, e, w);
                return;
            }
        }
    }

    free_back_data(data, network->count);
    free_back_data(expected, network->count);
    free_network(network);
    free_input_data(input);
    printf("traverse test OK!\n");
}

void alloc_flattened_test_data_test() {
    double inputs[] = {0, 7, 8, 9, 6, 2, 44, 5, 6, 3 , 4 , 2 ,2};
    double expected[] = {77, 88 ,6 ,2 ,5, 74,8 ,5 ,5 ,2, 7,4, 9 ,89, 5, 3 ,2 ,4, 7,7, 5,5};

    test_data* data = alloc_transfer_flattened_data(inputs, 2, expected, 3, 5);

    for (int i = 0; i < 5; i++) {
        int start = i * 2;
        for (int j = 0; j < 2; j++) {
            if (!def_deq(data->inputs[i].values[j], inputs[start + j])) printf("Bad correspondance\n");
        }

        start = i * 3;
        for (int j = 0; j < 3; j++) {
            if (!def_deq(data->expected[i].values[j], expected[start + j])) printf("Bad correspondance\n");
        }
    }

    free_transferred_flattened_data(data);
    data = alloc_transfer_flattened_data(big_arr1, 7, big_arr2, 3, 128);

    for(int i = 0; i < data->count; i++) {
        int total = 0;
        input_data current = data->inputs[i];
        for(int j = 0; j < current.count; j++) {
            total += (int)current.values[j];
        }

        double* ex = data->expected[i].values;
        double in[3];

        if (total >= 4) in[0] = 1;
        else in[0] = 0;
        if (total == 2 || total == 3 || total == 6 || total == 7) in[1] = 1;
        else in[1] = 0;
        if (total % 2 == 1) in[2] = 1;
        else in[2] = 0;

        for(int j = 0; j < 3; j++) {
            if (in[j] != ex[j]) {
                printf("BAD\n");
            }
        }
    }

    free_transferred_flattened_data(data);
    printf("alloc flattened test data OK!\n");
}

void repository_test() {
    const char filename[] = "neural_network_repository_test.nn";

    const int numbers[] = {784, 200, 100, 10};
    neural_network* network = alloc_network(4, numbers);
    if (save(network, filename) != 0) {
        printf("Save failed");
        return;
    }
    neural_network* download = restore(filename);

    if(network->count != download->count) {
        printf("Different network count");
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

        const int in = network->layers[i].in_count;
        const int out = network->layers[i].out_count;

        for(int o = 0; o < out; o++) {
            if(!def_deq(network->layers[i].biases[o], download->layers[i].biases[o])) {
                printf("Different bias for layer %d and output %d", i, o);
                return;
            }

            for(int j = 0; j < in; j++) {
                if(!def_deq(network->layers[i].weights[j * out + o], download->layers[i].weights[j * out + o])) {
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
    init_random();

    alloc_flattened_test_data_test();
    generate_test(0);
    traverse_test();
    repository_test();
}

