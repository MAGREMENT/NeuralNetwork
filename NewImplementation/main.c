#include <stdio.h>
#include <stdlib.h>

#include "conv_layer.h"
#include "utils.h"

void unit_test();

int main(void) {
    unit_test();
    return EXIT_SUCCESS;
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
    conv_layer_forward_test();
}
