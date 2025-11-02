#include <stdio.h>
#include <stdlib.h>

#include "builder.h"
#include "Layers/Types/convolutional_layer.h"
#include "neural_network.h"
#include "utils.h"
#include "Layers/Types/activation_layer.h"
#include "Layers/Types/dense_layer.h"
#include "Optimizers/Types/gradient_descent_optimizer.h"
#include "Util/math_util.h"

double big_arr1[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1};

double big_arr2[] = {0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1};

void unit_test();

int main(void) {
    unit_test();
    return EXIT_SUCCESS;
}

typedef struct bal_b {
    int opt;
    optimizer_cnstr_args opt_args;
    double learning_rate;
} bal_b;

void bit_add_learn_test(bool verbose) {
    const bal_b builds[] = {
        {GRADIENT_DESCENT, (optimizer_cnstr_args) {.value = 0}, 1},
        {MOMENTUM_GRADIENT_DESCENT, (optimizer_cnstr_args) {.value = 0.1}, 1},
        {NESTEROV, (optimizer_cnstr_args) {.value = 0.9}, 1},
        //{ADAM, (optimizer_cnstr_args) {.values = (double2) {0.9, 0.999}}, 1},
    };

    builder* b = alloc_builder(7);
    b_dense(b, 4);
    b_activation(b, SIGMOID);
    b_dense(b, 3);
    b_activation(b, SIGMOID);

    b_ds(b, FULL_BATCH, (data_selector_cnstr_args)0);
    b_sch(b, CONSTANT, (scheduler_cnstr_args)0.0);

    b->cost_type = BINARY_CROSS_ENTROPY;

    test_data test;
    test.count = 128;
    test.inputs = big_arr1;
    test.expected = big_arr2;

    for (int i = 0; i < sizeof(builds) / sizeof(bal_b); i++) {
        b_opt(b, builds[i].opt, builds[i].opt_args);
        b->learningRate = builds[i].learning_rate;

        neural_network* n = build(b);
        initialize(n);

        set_all_weights_and_biases(n->layers[0], 1, 1);
        set_all_weights_and_biases(n->layers[2], 1, 1);

        learning_state* state = alloc_state(n);

        double cost = get_avg_cost(n, test);

        for (int j = 0; j < 10; j++) {
            iterative_learn(n, test, state, 100);

            const double buffer = get_avg_cost(n, test);
            if (buffer >= cost) {
                printf("bit learn cost fail for optimizer %d and iteration %d\n", i, j * 10);
                return;
            }

            cost = buffer;
        }

        if (verbose) {
            printf("Cost for optimizer %d : %f\n", i, cost);
        }

        free_state(n, state);
        free_neural_network(n, 1);
    }

    free_builder(b);
    printf("bit add learn testr OK!\n");
}

void build_test() {
    builder* b = alloc_builder(9);

    b_dense(b, 7);
    b_activation(b, LEAKY_RELU);
    b_activation(b, SIGMOID);
    b_dense(b, 3);
    b_activation(b, SOFTMAX);

    b_opt(b, NESTEROV, (optimizer_cnstr_args) {.value = 0.9});
    b_ds(b, FULL_BATCH, (data_selector_cnstr_args)0);
    b_sch(b, COSINE_DECAY, (scheduler_cnstr_args) {.di_value = (double_int) {0.5, 20}});

    b->cost_type = MEAN_SQUARE;

    neural_network* n = build_free(b);

    if (n->layerCount != 5) {
        printf("Wrong layer count\n");
        return;
    }

    if (n->layers[0]->in_count != 9 || n->layers[0]->out_count != 7 ||
        n->layers[1]->in_count != 7 || n->layers[1]->out_count != 7 ||
        n->layers[2]->in_count != 7 || n->layers[2]->out_count != 7 ||
        n->layers[3]->in_count != 7 || n->layers[3]->out_count != 3 ||
        n->layers[4]->in_count != 3 || n->layers[4]->out_count != 3) {
        printf("Wrong layer i/o\n");
        return;
    }

    if (n->layers[0]->gradient_count != 9 * 7 + 7 ||
        n->layers[1]->gradient_count != 0 ||
        n->layers[2]->gradient_count != 0 ||
        n->layers[3]->gradient_count != 7 * 3 + 3 ||
        n->layers[4]->gradient_count != 0) {
        printf("Wrong layer gradient count\n");
        return;
    }

    if (n->layers[0]->initialize != initialize_dense_he
        || n->layers[3]->initialize != initialize_dense_random) {
        printf("Wrong dense layer initialize func\n");
        return;
    }

    //TODO more tests

    free_neural_network(n, true);
    printf("builder test OK!\n");
}

void dense_test() {
    neural_network* n = alloc_neural_network(2);
    n->layers[0] = cnstr_dense_layer(2, 3, initialize_dense_to_one);
    n->layers[1] = cnstr_dense_layer(3, 2, initialize_dense_to_one);

    double w1[] = {0.5, 1, 1.5, 0.5, 1, 1.5};
    double w2[] = {1.5, 1, 0.5, 1.5, 0.5, 1};
    double b1[] = {-1, 0, -1};
    double b2[] = {-2, -2};

    set_weights(n->layers[0], w1);
    set_weights(n->layers[1], w2);
    set_biases(n->layers[0], b1);
    set_biases(n->layers[1], b2);

    double i[] = {2, 2};
    double o[3];

    n->layers[0]->vtable->forward(n->layers[0], i, o);

    if (!def_deq(o[0], 1)) {
        printf("DENSE TEST FAIL !\n");
        return;
    }
    if (!def_deq(o[1], 4)) {
        printf("DENSE TEST FAIL !\n");
        return;
    }
    if (!def_deq(o[2], 5)) {
        printf("DENSE TEST FAIL !\n");
        return;
    }

    double o2[2];

    n->layers[1]->vtable->forward(n->layers[1], o, o2);

    if (!def_deq(o2[0], 4)) {
        printf("DENSE TEST FAIL !\n");
        return;
    }
    if (!def_deq(o2[1], 10)) {
        printf("DENSE TEST FAIL !\n");
        return;
    }

    double o3[2];

    predict(n, i, o3);

    if (!def_deq(o2[0], o3[0])) {
        printf("DENSE TEST FAIL !\n");
        return;
    }
    if (!def_deq(o2[1], o3[1])) {
        printf("DENSE TEST FAIL !\n");
        return;
    }

    n->cost_vtable = cost_vtables + MEAN_SQUARE;
    double e[] = {2, 7};

    double cost = get_cost(n, i, e);

    if (!def_deq(cost, 4 + 9)) {
        printf("DENSE TEST FAIL !\n");
        return;
    }

    n->optimizer = cnstr_gradient_descent_optimizer(); //TODO to set_optimizer
    const learning_args args = {0.001, NULL};

    for (int epoch = 0; epoch < 10; epoch++) {

        learn(n, (test_data){i, e, 1}, (range){1, 0, 1}, args);
        const double cost2 = get_cost(n, i, e);

        if (cost2 >= cost) {
            printf("Dense test cost lowering fail\n");
            return;
        }

        cost = cost2;
    }

    free_neural_network(n, true);
    printf("dense test OK!\n");
}

void predict_test() {
    const double biases = 1;
    const double weights = 1;
    const double i1 = 1;
    const double i2 = 2;

    neural_network* n = alloc_neural_network(4);
    n->layers[0] = cnstr_dense_layer(2, 3, initialize_dense_to_one);
    n->layers[1] = cnstr_activation_layer(SIGMOID, 3);
    n->layers[2] = cnstr_dense_layer(3, 2, initialize_dense_to_one);
    n->layers[3] = cnstr_activation_layer(SIGMOID, 2);
    initialize(n);

    const double inputs[] = {i1, i2};
    double outputs[2];

    double buffer = biases + weights * i1 + weights * i2;
    buffer = sigmoid(buffer);
    buffer = biases + buffer * weights * 3;
    buffer = sigmoid(buffer);

    predict(n, inputs, outputs);

    for (int i = 0; i < 2; i++) {
        if (!def_deq(buffer, outputs[i])) {
            printf("Wrong predict output\n");
        }
    }

    free_neural_network(n, 1);
    printf("predict test OK!\n");
}

void conv_layer_forward_test() {
    const size3D is = {3, 3, 1};
    const size2D ks = {2, 2};
    layer* l = cnstr_conv_layer(is, ks,1, 1, 0);
    set_kernels_and_biases(l, 0, 0);

    conv_layer_params* p = l->params;
    p->kernels[0] = 1;
    p->kernels[1] = 2;
    p->kernels[2] = -1;
    p->kernels[3] = 0;

    if (p->output_size.width != 2 || p->output_size.height != 2 || p->output_size.depth != 1) {
        printf("Wrong output size\n");
        return;
    }

    double input[] = {1, 6, 2, 5, 3, 1, 7, 0, 4};
    double expected[] = {8, 7, 4, 5};
    double result[4];

    l->vtable->forward(l, input, result);

    for (int i = 0; i < 4; i++) {
        if (!def_deq(result[i], expected[i])) {
            printf("Wrong output value at index %i", i);
            return;
        }
    }

    l->vtable->free(l);
    printf("conv layer forward test OK!\n");
}

void unit_test() {
    bit_add_learn_test(false);
    build_test();
    dense_test();
    predict_test();
    conv_layer_forward_test();
}
