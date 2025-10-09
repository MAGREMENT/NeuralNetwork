#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "neural_network.h"
#include "big-array.c"
#include "functions.h"

void unit_tests();
void cut_2D_test();

int main() {
    //cut_2D_test();
    //unit_tests();

    //return EXIT_SUCCESS;

    clock_t start = clock();

    const int numbers[] = {7, 4, 3};
    neural_network* network = alloc_network(3, numbers);
    apply_default_hyper_params(network);
    set_activation_type(network, SIGMOID, SIGMOID);
    network->learningRate = 10;
    network->threadCount = 1;

    //test_data* data = positive_generate_for_2D(1, 100, 2, parable_10_cut);
    test_data* data = alloc_flattened_test_data(big_arr1, 7, big_arr2, 3, 128);

    standardize(data);

    initialize(network);

    gradient_diagnostic* diag = alloc_run_gradient_diagnostic(network, data, -5, 5);
    print_diagnostic(network, diag);

    iterative_learn(network, data, NULL, 10000);

    printf("%f\n", avg_cost(network, data));

    double accuracy = 0;
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

    printf("%f\n", accuracy / data->count * 100);

    free_gradient_diagnostic(diag);
    diag = alloc_run_gradient_diagnostic(network, data, -5, 5);
    print_diagnostic(network, diag);

    free_gradient_diagnostic(diag);
    free_network(network);
    free_test_data(data);

    clock_t end = clock();

    printf("Time : %fs", (double)(end - start) / CLOCKS_PER_SEC);

    return EXIT_SUCCESS;
}

/*neural_network* alloc_example_network(int activation) {
    int numbers[] = {2, 3, 2};
    neural_network* network = alloc_network(3, numbers);
    params params;
    params.initialLearningRate = 1;
    params.activationType = activation;
    params.outputActivationType = activation;
    params.costType = MEAN_SQUARED;
    apply_params(network, params);

    return network;
}

neural_network* alloc_example_network_with_data(int activation) {
    neural_network* network = alloc_example_network(activation);

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

void learn_test() {
    test_data *test = positive_generate_for_2D(0.5, 20, 2, diagonal_cut);
    neural_network* networks[2] = {
        alloc_example_network_with_data(SIGMOID),
        alloc_example_network_with_data(SIGMOID)
    };

    set_activation_type(networks[1], RELU, SOFTMAX);
    set_cost_type(networks[1], CROSS_ENTROPY);

    for (int i = 0; i < 2; i++) {
        neural_network* network = networks[i];

        double cost = avg_cost(network, test);
        for(int i = 0; i < 1000; i++) {
            learn(network, test, 0, test->count, network->learningRate / test->count, NULL);
            double nCost = avg_cost(network, test);

            if(nCost > cost) {
                printf("new cost (%.5f) bigger than previous cost (%.5f) on iteration %d\n", nCost, cost, i);
                return;
            }

            cost = nCost;
        }

        free_network(network);
    }

    free_test_data(test);
    printf("learn test OK!\n");
}

void cost_test() {
    neural_network* network = alloc_example_network_with_data(DEFAULT);
    input_data* input = alloc_input_data(2);
    input_data* expected = alloc_input_data(2);
    input->values[0] = 1;
    input->values[1] = 1;
    expected->values[0] = 1;
    expected->values[1] = 1;

    cost(network, input, expected);
    //TODO make for real

    free_network(network);
    free_input_data(input);
    free_input_data(expected);
}

void repository_test() {
    constexpr char filename[] = "neural_network_repository_test.nn";
    FILE* fptr = fopen(filename, "w");
    fclose(fptr);

    neural_network* network = alloc_example_network_with_data(SIGMOID);
    params params;
    params.initialLearningRate = 1;
    params.regularization = 0.02;
    params.momentum = 0.9;
    params.learningRateDecay = 0.05;
    params.activationType = SIGMOID;
    params.outputActivationType = SIGMOID;
    params.costType = MEAN_SQUARED;
    apply_params(network, params);

    save(network, &params, filename);
    neural_network* download = restore(filename, NULL);

    if(network->count != download->count) {
        printf("Different network count");
        return;
    }

    if(network->learningRate != download->learningRate) {
        printf("Different learning rate");
        return;
    }

    if(network->learningRateDecay != download->learningRateDecay) {
        printf("Different learning rate decay");
        return;
    }

    if(network->regularization != download->regularization) {
        printf("Different regularization");
        return;
    }

    if(network->momentum != download->momentum) {
        printf("Different momentum");
        return;
    }

    if(network->cost != download->cost) {
        printf("Different cost function");
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

void alloc_flattened_test_data_test() {
    double inputs[] = {0, 7, 8, 9, 6, 2, 44, 5, 6, 3 , 4 , 2 ,2};
    double expected[] = {77, 88 ,6 ,2 ,5, 74,8 ,5 ,5 ,2, 7,4, 9 ,89, 5, 3 ,2 ,4, 7,7, 5,5};

    test_data* data = alloc_flattened_test_data(inputs, 2, expected, 3, 5);

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

    free_test_data(data);
    data = alloc_flattened_test_data(big_arr1, 7, big_arr2, 3, 128);

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

    free_test_data(data);
    printf("alloc flattened test data OK!\n");
}

void full_test_value_check(double v, double expected) {
    if (!deq(v, expected, 0.01)) {
        printf("FULL TEST FAIL !!!!!!!!!!!!!!!!!\n");
    }
}

void full_test() {
    neural_network* n = alloc_example_network_with_data(DEFAULT);

    input_data i;
    double iv[] = {2, 2};
    i.count = 2;
    i.values = iv;

    input_data o1;
    double ov[3];
    o1.count = 3;
    o1.values = ov;

    forward(n->layers[0], i, &o1);

    full_test_value_check(o1.values[0], 1);
    full_test_value_check(o1.values[1], 4);
    full_test_value_check(o1.values[2], 5);

    input_data o2;
    double ov2[2];
    o2.count = 2;
    o2.values = ov2;

    forward(n->layers[1], o1, &o2);

    full_test_value_check(o2.values[0], 4);
    full_test_value_check(o2.values[1], 10);

    input_data o3;
    double ov3[2];
    o3.count = 2;
    o3.values = ov3;

    predict(n, &i, &o3);

    full_test_value_check(o2.values[0], o3.values[0]);
    full_test_value_check(o2.values[1], o3.values[1]);

    backpropagation_data* back = alloc_traverse(n, &i);

    full_test_value_check(o1.values[0], back[0].afterActivations[0]);
    full_test_value_check(o1.values[1], back[0].afterActivations[1]);
    full_test_value_check(o1.values[2], back[0].afterActivations[2]);
    full_test_value_check(o2.values[0], back[1].afterActivations[0]);
    full_test_value_check(o2.values[1], back[1].afterActivations[1]);

    full_test_value_check(1, back[0].weightedInputs[0]);
    full_test_value_check(4, back[0].weightedInputs[1]);
    full_test_value_check(5, back[0].weightedInputs[2]);
    full_test_value_check(4, back[1].weightedInputs[0]);
    full_test_value_check(10, back[1].weightedInputs[1]);

    layer_data* gradients = alloc_layer_data_array(n, 0);
    input_data e;
    double ev[] = {-2, -2};
    e.count = 2;
    e.values = ev;

    update_gradients(n, gradients, i, e);
    //Node values :
    //6 * 1 = 12
    //12 * 1 = 24
    const double nv0 = 6;
    const double nv1 = 12;

    full_test_value_check(gradients[1].weights[0], nv0 * 1);
    full_test_value_check(gradients[1].weights[1], nv1 * 1);
    full_test_value_check(gradients[1].weights[2], nv0 * 4);
    full_test_value_check(gradients[1].weights[3], nv1 * 4);
    full_test_value_check(gradients[1].weights[4], nv0 * 5);
    full_test_value_check(gradients[1].weights[5], nv1 * 5);

    full_test_value_check(gradients[1].biases[0], nv0);
    full_test_value_check(gradients[1].biases[1], nv1);

    //Node values :
    //9 + 12 = 21
    //3 + 18 = 21
    //3 + 12 = 15;
    const double nv2 = 21;
    const double nv3 = 21;
    const double nv4 = 15;

    full_test_value_check(gradients[0].weights[0], nv2 * 2);
    full_test_value_check(gradients[0].weights[1], nv3 * 2);
    full_test_value_check(gradients[0].weights[2], nv4 * 2);
    full_test_value_check(gradients[0].weights[3], nv2 * 2);
    full_test_value_check(gradients[0].weights[4], nv3 * 2);
    full_test_value_check(gradients[0].weights[5], nv4 * 2);

    full_test_value_check(gradients[0].biases[0], nv2);
    full_test_value_check(gradients[0].biases[1], nv3);
    full_test_value_check(gradients[0].biases[2], nv4);

    n->learningRate = 0.001;
    const double c1 = cost(n, &i, &e);

    apply_gradients(n->layers[1], gradients[1], n->learningRate);
    apply_gradients(n->layers[0], gradients[0], n->learningRate);

    full_test_value_check(n->layers[0].weights[0], 0.5 - nv2 * 2 * n->learningRate);

    const double c2 = cost(n, &i, &e);

    if (c2 > c1) printf("Problem with cost");

    test_data test;
    test.count = 1;
    test.inputs = &i;
    test.expected = &e;

    free_network(n);
    n = alloc_example_network_with_data(DEFAULT);
    n->learningRate = 0.001;
    linear_batch_learn(n, &test, NULL, 1, 1000);

    neural_network* n2 = alloc_example_network_with_data(DEFAULT);
    n2->learningRate = 0.001;

    learning_state* state = alloc_state(n);
    for (int i = 0; i < 100; i++) {
        linear_batch_learn(n2, &test, state, 1, 10);
    }
    free_state(state, n->count);

    const double c3 = cost(n, &i, &e);
    if (c3 > c1) printf("Problem with cost");

    input_data o4;
    double ov4[2];
    o4.count = 2;
    o4.values = ov4;

    input_data o5;
    double ov5[2];
    o5.count = 2;
    o5.values = ov5;

    predict(n, &i, &o4);
    predict(n, &i, &o5);

    full_test_value_check(ov4[0], ev[0]);
    full_test_value_check(ov4[1], ev[1]);

    full_test_value_check(ov4[0], ov5[0]);
    full_test_value_check(ov4[1], ov5[1]);

    free_back_data(back, n->count);
    free_network(n);
    printf("full test OK!\n");
}

void predict_test() {
    int numbers[] = {7, 5, 4, 3};
    neural_network* network = alloc_network(4, numbers);
    params params;
    params.initialLearningRate = 1;
    params.learningRateDecay = 0;
    params.regularization = 0.1;
    params.momentum = 0.9;
    params.activationType = SIGMOID;
    params.outputActivationType = SIGMOID;
    params.costType = MEAN_SQUARED;
    apply_params(network, params);
    initialize(network);

    double iv[] = {1, 0, 1, 1, 1, 0, 0};
    input_data i;
    i.values = iv;
    i.count = 7;

    double ov[3];
    input_data o;
    o.values = ov;
    o.count = 3;

    predict(network, &i, &o);

    free_network(network);
    printf("predict test OK!\n");
}

void are_networks_same(neural_network* n1, neural_network* n2) {
    for (int l = 0; l < n1->count; l++) {
        const int oc = n1->layers[l].out_count;
        for (int o = 0; o < oc; o++) {
            for (int i = 0; i < n1->layers[l].in_count; i++) {
                const int index = i * oc + o;

                const double w1 = n1->layers[l].weights[index];
                const double w2 = n2->layers[l].weights[index];
                if (w1 != w2) {
                    printf("BAD\n");
                }
            }

            const double b1 = n1->layers[l].biases[o];
            const double b2 = n2->layers[l].biases[o];
            if (b1 != b2) {
                printf("BAD\n");
            }
        }
    }
}

void learn_deterministic_test() {
    int types[] = {SIGMOID};
    for (int i = 0; i < 1; i++) {
        const int type = types[i];

        neural_network* n1 = alloc_example_network_with_data(type);
        neural_network* n2 = alloc_example_network_with_data(type);
        n1->learningRate = 0.0001;
        n2->learningRate = 0.0001;

        test_data d1;
        d1.count = 1;

        input_data i1;
        double iv1[] = {2, 2};
        i1.count = 2;
        i1.values = iv1;

        input_data e1;
        double ev1[] = {-2, -2};
        e1.count = 2;
        e1.values = ev1;

        d1.inputs = &i1;
        d1.expected = &e1;

        learn(n1, &d1, 0, 1, 1, NULL);
        learn(n2, &d1, 0, 1, 1, NULL);

        are_networks_same(n1, n2);

        linear_batch_learn(n1, &d1, NULL, 1, 100);
        linear_batch_learn(n2, &d1, NULL, 1, 100);

        are_networks_same(n1, n2);
    }

    printf("Learn deterministic test OK ! \n");
}

void unit_tests() {
    init_random();

    full_test();

    predict_test();
    learn_deterministic_test();
    generate_test(0);
    learn_test();
    cost_test();
    traverse_test();
    repository_test();
    alloc_flattened_test_data_test();
}*/

