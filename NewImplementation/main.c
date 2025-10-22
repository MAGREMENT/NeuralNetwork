#include <stdio.h>
#include <stdlib.h>

#include "conv_layer.h"
#include "neural_network.h"
#include "utils.h"
#include "Layers/Types/activation_layer.h"
#include "Layers/Types/dense_layer.h"
#include "Util/string_math.h"

void unit_test();

int main(void) {
    unit_test();
    return EXIT_SUCCESS;
}

void predict_test() {
    const double biases = 1;
    const double weights = 1;
    const double i1 = 1;
    const double i2 = 2;

    //TODO nn builder
    layer* layers[] = {cnstr_dense_layer(2, 3, initialize_dense_to_one),
        cnstr_activation_layer(SIGMOID, 3),
        cnstr_dense_layer(3, 2, initialize_dense_to_one),
        cnstr_activation_layer(SIGMOID, 2)};

    neural_network* n = alloc_neural_network(4, layers);
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

    printf("predict test OK!\n");
}

void conv_layer_forward_test() {
    const size3D is = {3, 3, 1};
    const size3D ks = {2, 2, 1};
    conv_layer* l = alloc_conv_layer(is, ks,1, 1, 0);
    set_kernels_and_biases(l, 0, 0);

    l->kernels[0] = 1;
    l->kernels[1] = 2;
    l->kernels[2] = -1;
    l->kernels[3] = 0;

    if (l->output_size.width != 2 || l->output_size.height != 2 || l->output_size.depth != 1) {
        printf("Wrong output size\n");
        return;
    }

    double input[] = {1, 6, 2, 5, 3, 1, 7, 0, 4};
    double expected[] = {8, 7, 4, 5};

    double* result = conv_forward(l, input);

    for (int i = 0; i < 4; i++) {
        if (!def_deq(result[i], expected[i])) {
            printf("Wrong output value at index %i", i);
            return;
        }
    }

    printf("conv layer forward test OK!\n");
}

void unit_test() {
    predict_test();
    conv_layer_forward_test();
}
