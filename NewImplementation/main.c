#include <stdio.h>
#include <stdlib.h>

#include "Layers/Types/convolutional_layer.h"
#include "neural_network.h"
#include "utils.h"
#include "Layers/Types/activation_layer.h"
#include "Layers/Types/dense_layer.h"
#include "Optimizers/GradientDescent/gradient_descent_optimizer.h"
#include "Util/math_util.h"

void unit_test();

int main(void) {
    unit_test();
    return EXIT_SUCCESS;
}

void full_test_value_check(const double v, const double expected) {
    if (!deq(v, expected, 0.01)) {
        printf("FULL TEST FAIL !\n");
    }
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

    n->layers[0]->functions.forward(n->layers[0], i, o);

    full_test_value_check(o[0], 1);
    full_test_value_check(o[1], 4);
    full_test_value_check(o[2], 5);

    double o2[2];

    n->layers[1]->functions.forward(n->layers[1], o, o2);

    full_test_value_check(o2[0], 4);
    full_test_value_check(o2[1], 10);

    double o3[2];

    predict(n, i, o3);

    full_test_value_check(o2[0], o3[0]);
    full_test_value_check(o2[1], o3[1]);

    n->cost_vtable = cost_vtables + MEAN_SQUARE;
    double e[] = {2, 7};

    double cost = get_cost(n, i, e);

    full_test_value_check(cost, 4 + 9);

    n->optimizer = cnstr_gradient_descent_optimizer(); //TODO to set_optimizer
    const optimizer_args args = {0.001};

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

    l->functions.forward(l, input, result);

    for (int i = 0; i < 4; i++) {
        if (!def_deq(result[i], expected[i])) {
            printf("Wrong output value at index %i", i);
            return;
        }
    }

    l->functions.free(l);
    printf("conv layer forward test OK!\n");
}

void unit_test() {
    dense_test();
    predict_test();
    conv_layer_forward_test();
}
