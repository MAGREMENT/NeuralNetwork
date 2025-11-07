#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "builder.h"
#include "Layers/Types/convolutional_layer.h"
#include "neural_network.h"
#include "Layers/Types/activation_layer.h"
#include "Layers/Types/dense_layer.h"
#include "Layers/Types/pooling_layer.h"
#include "Optimizers/Types/simple_optimizer.h"
#include "Util/double_util.h"
#include "Util/math_util.h"
#include "Util/rand_util.h"

#ifdef _MSC_VER

#include "Layers/Types/Cuda/cuda_dense_layer.cuh"

#endif

double big_arr1[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1};

double big_arr2[] = {0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1};

void mnist_run();
void unit_test();

int main(void) {
    //mnist_run();
    unit_test();
    return EXIT_SUCCESS;
}

//TODO fix somehow
void mnist_run() {
    //max : 10000
    const int count = 100;

    const int iCount = 784 * count;
    const int lCount = 10 * count;

    double* images = malloc(sizeof(double) * iCount);
    double* labels = malloc(sizeof(double) * lCount);

    FILE* iFile = fopen("Data/images.data", "r");
    fread(images, sizeof(double), iCount, iFile);
    fclose(iFile);

    FILE* oFile = fopen("Data/labels.data", "r");
    fread(labels, sizeof(double), lCount, oFile);
    fclose(oFile);

    test_data test = {images, labels, count};
    builder* b = alloc_builder(784);

    b_dense(b, 200);
    b_activation(b, RELU);
    b_dense(b, 100);
    b_activation(b, RELU);
    b_dense(b, 10);
    b_activation(b, SOFTMAX);

    b_opt(b, SIMPLIFIED_NESTEROV, (optimizer_cnstr_args) {.value = 0.8});
    b_sch(b, CONSTANT, (scheduler_cnstr_args) {.value = 0.0});
    b_ds(b, MINI_BATCH, (data_selector_cnstr_args) {.value = 64});

    b->cost_type = BINARY_CROSS_ENTROPY;

    b->shuffleDataOnIteration = 1;
    b->learningRate = 0.1;

    neural_network* n = build_free(b);
    initialize(n);

    printf("Iteration 0 : Cost -> %f | Accuracy -> %f\n", get_avg_cost(n, test), get_classification_accuracy(n, test));
    learning_state* state = alloc_state(n);

    for (int i = 0; i < 10; i++) {
        iterative_learn(n, test, state, 1);
        printf("Iteration %d : Cost -> %f | Accuracy -> %f\n", i + 1, get_avg_cost(n, test), get_classification_accuracy(n, test));
    }

    free_neural_network(n, 1);
    free(images);
    free(labels);
}

void pooling_layer_test() {
    const double i1[] = {1, 9, 3, 4, 5, 10, 12, 4, 1};
    const double i2[] = {1, 2, 3, 8, 9, 6, 5, 4, 2, 1, 3, 4, -3, 2, 7, 8};
    const double i3[] = {1, 5, 7, 8, 14, -1, 3, 2};

    layer* l = cnstr_pooling_layer(POOLING_MAX, (size3D){3, 3, 1}, (size2D) {2, 2}, 1, 0);

    double o1[4];
    const double e1[] = {9, 10, 12, 10};

    l->vtable->forward(l, i1, o1);
    for (int i = 0; i < 4; i++) {
        if (!def_deq(o1[i], e1[i])) {
            printf("Forward 1 fail\n");
            return;
        }
    }

    l->vtable->free(l);

    l = cnstr_pooling_layer(POOLING_MAX, (size3D){4, 4, 1}, (size2D) {2, 2}, 2, 1);

    double o2[9];
    const double e2[] = {1, 3, 8, 9, 6, 4, -3, 7, 8};

    l->vtable->forward(l, i2, o2);
    for (int i = 0; i < 9; i++) {
        if (!def_deq(o2[i], e2[i])) {
            printf("Forward 2 fail\n");
            return;
        }
    }

    l->vtable->free(l);

    //TODO backward & i3 & avg

    printf("pooling layer test OK!\n");
}

typedef struct bal_b {
    int opt;
    optimizer_cnstr_args opt_args;
    double learning_rate;
} bal_b;

static void get_simple_cost_arr(neural_network* dummy, const double* in, double* predicted, const double* expected, double* cost) {
    predict(dummy, in, predicted);
    for (int i = 0; i < 4; i++) {
        cost[i] = predicted[i] - expected[i];
    }
}

static void print_costs(double* costs, int count) {
    if (count == 0) return;

    printf("%.4f", costs[0]);

    for (int i = 1; i < count; i++) {
        printf(" | %.4f", costs[i]);
    }

    printf("\n");
}

void optimizer_test(const int error, const int verbose) {
    const bal_b builds[] = {
        {SIMPLE, (optimizer_cnstr_args) {.value = 0}, 0.1},
        {FREE_MOMENTUM, (optimizer_cnstr_args) {.value = 0.2}, 0.1},
        {PROPORTIONAL_MOMENTUM, (optimizer_cnstr_args) {.value = 0.9}, 0.1},
        {SIMPLIFIED_NESTEROV, (optimizer_cnstr_args) {.value = 0.3}, 0.1},
        {RMS_PROP, (optimizer_cnstr_args) {.value = 0.9}, 0.1},
        {ADAM, (optimizer_cnstr_args) {.values = (double2) {0.9, 0.999}}, 0.1},
        {ADAGRAD, (optimizer_cnstr_args) {.value = 0.0}, 0.1},
        {ADADELTA, (optimizer_cnstr_args) {.value = 0.9}, 0.1}
    };

    neural_network* dummy = alloc_neural_network(1);
    layer* l = cnstr_dense_layer(1, 4, initialize_dense_to_one);
    dummy->layers[0] = l;
    initialize(dummy);

    dense_layer_params* p = l->params;

    double in[] = {1};
    double expected[] = {-7, 15, 0, 3};
    double predicted[4];
    double cost[4];

    for (int i = 0; i < sizeof(builds) / sizeof(bal_b); ++i) {
        if (verbose) printf("Optimizer %d : \n", i);

        p->weights[0] = 1;
        p->weights[1] = 1;
        p->weights[2] = 1;
        p->weights[3] = 1;

        optimizer* opt = cnstr_optimizer(builds[i].opt, builds[i].opt_args);
        void* state = opt->vtable->cnstr_state(opt, dummy);
        optimizer_args args = {builds[i].learning_rate, 0, 0, state};

        get_simple_cost_arr(dummy, in, predicted, expected, cost);
        if (verbose) print_costs(cost, 4);

        for (int iteration = 0; iteration < 10; iteration++) {
            args.iteration += 1;
            opt->vtable->apply_gradients(opt, p->weights, cost, 4, args);

            if (verbose) printf("Iteration %d : \n", iteration + 1);

            double buffer[4];
            get_simple_cost_arr(dummy, in, predicted, expected, buffer);

            if (error) {
                for (int j = 0; j < 4; j++) {
                    if (fabs(buffer[j]) > fabs(cost[j])) {
                        printf("Optimizer %d : Fail\n", i);
                        return;
                    }
                }
            }

            memcpy(cost, buffer, 4 * sizeof(double));
            if (verbose) print_costs(cost, 4);
        }

        opt->vtable->free_state(state, dummy);
        opt->vtable->free(opt);

        if (verbose) printf("\n");
    }

    free_neural_network(dummy, 1);
    printf("optimizer test OK!\n");
}

void bit_add_learn_test(const int verbose) {
    const bal_b builds[] = {
        {SIMPLE, (optimizer_cnstr_args) {.value = 0}, 1},
        {FREE_MOMENTUM, (optimizer_cnstr_args) {.value = 0.1}, 1},
        {PROPORTIONAL_MOMENTUM, (optimizer_cnstr_args) {.value = 0.9}, 1},
        {SIMPLIFIED_NESTEROV, (optimizer_cnstr_args) {.value = 0.8}, 1},
        {RMS_PROP, (optimizer_cnstr_args) {.value = 0.9}, 0.1},
        {ADAM, (optimizer_cnstr_args) {.values = (double2) {0.9, 0.999}}, 1},
        {ADAGRAD, (optimizer_cnstr_args) {.value = 0.0}, 1},
        {ADADELTA, (optimizer_cnstr_args) {.value = 0.9}, 1}
    };

    builder* b = alloc_builder(7);
    b_dense(b, 4);
    b_activation(b, SIGMOID);
    b_dense(b, 3);
    b_activation(b, SIGMOID);

    b_ds(b, FULL_BATCH, (data_selector_cnstr_args) {.value = 0});
    b_sch(b, CONSTANT, (scheduler_cnstr_args) {.value = 0.0});

    b->cost_type = BINARY_CROSS_ENTROPY;
    b->shuffleDataOnIteration = 0;

    test_data test;
    test.count = 128;
    test.inputs = big_arr1;
    test.expected = big_arr2;

    for (int i = 0; i < sizeof(builds) / sizeof(bal_b); i++) {
        b_opt(b, builds[i].opt, builds[i].opt_args);
        b->learningRate = builds[i].learning_rate;

        neural_network* n = build(b);
        initialize(n);

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
            printf("Optimizer %d -> Cost : %f | Accuracy = %f\n", i, cost, get_binary_accuracy(n, test));
        }

        free_state(n, state);
        free_neural_network(n, 1);
    }

    free_builder(b);
    printf("bit add learn test OK!\n");
}

void build_test() {
    builder* b = alloc_builder(9);

    b_dense(b, 7);
    b_activation(b, LEAKY_RELU);
    b_activation(b, SIGMOID);
    b_dense(b, 3);
    b_activation(b, SOFTMAX);

    b_opt(b, SIMPLIFIED_NESTEROV, (optimizer_cnstr_args) {.value = 0.9});
    b_ds(b, FULL_BATCH, (data_selector_cnstr_args) {.value = 0});
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

    free_neural_network(n, 1);
    printf("builder test OK!\n");
}

void mt_dense_test(int verbose) {
    const int inCount = 10000;
    const int outCount = 10000;

    double* in = malloc(inCount * sizeof(double));
    double* out1 = malloc(outCount * sizeof(double));
    double* out2 = malloc(outCount * sizeof(double));
#ifdef _MSC_VER
    double* out3 = malloc(outCount * sizeof(double));
#endif

    for (int i = 0; i < inCount; i++) {
        in[i] = rand_d(-5, 5);
    }

    layer* single = cnstr_dense_layer(inCount, outCount, initialize_dense_to_zero);
    layer* multi = cnstr_multi_thread_dense_layer(inCount, outCount, 8, initialize_dense_to_zero);

    dense_layer_params* sp = single->params;
    dense_layer_params* mp = multi->params;

    clock_t singleTime = 0;
    clock_t multiTime = 0;

#ifdef _MSC_VER
    layer* cuda = cnstr_cuda_dense_layer(inCount, outCount, 256, initialize_dense_to_zero);
    dense_layer_params* cp = cuda->params;
    clock_t cudaTime = 0;
#endif

    for (int iteration = 0; iteration < 5; iteration++) {
        for (int i = 0; i < inCount * outCount; i++) {
            const double d = rand_d(-5, 5);
            sp->weights[i] = d;
            mp->weights[i] = d;
#ifdef _MSC_VER
            cp->weights[i] = d;
#endif
        }

        for (int i = 0; i < outCount; i++) {
            const double d = rand_d(-5, 5);
            sp->biases[i] = d;
            mp->biases[i] = d;
#ifdef _MSC_VER
            cp->biases[i] = d;
#endif
        }

        clock_t s = clock();
        single->vtable->forward(single, in, out1);
        clock_t e = clock();

        singleTime += e - s;

        s = clock();
        multi->vtable->forward(multi, in, out2);
        e = clock();

        multiTime += e - s;

#ifdef _MSC_VER
        s = clock();
        cuda->vtable->forward(cuda, in, out3);
        e = clock();

        cudaTime += e - s;
#endif

        for (int i = 0; i < outCount; i++) {
            if (!def_deq(out1[i], out2[i])) {
                printf("Not same value\n");
                goto end;
            }
        }

#ifdef _MSC_VER
        for (int i = 0; i < outCount; i++) {
            if (!def_deq(out1[i], out3[i])) {
                printf("Not same value\n");
                goto end;
            }
        }
#endif
    }

    if (verbose) {
        printf("Single thread time : %f s\n", (double)singleTime / CLOCKS_PER_SEC);
        printf("Multi thread time : %f s\n", (double)multiTime / CLOCKS_PER_SEC);
#ifdef _MSC_VER
        printf("GPU thread time : %f s\n", (double)cudaTime / CLOCKS_PER_SEC);
#endif
    }

    end:

    printf("multi-thread dense layer test OK!\n");
    free(single);
    free(multi);
#ifdef _MSC_VER
    free(cuda);
    free(out3);
#endif
    free(in);
    free(out1);
    free(out2);
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

    n->optimizer = cnstr_simple_optimizer(); //TODO to set_optimizer
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

    free_neural_network(n, 1);
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
    size3D is = {3, 3, 1};
    size2D ks = {2, 2};
    layer* l = cnstr_conv_layer(is, ks,1, 1, 0);

    set_kernels_and_biases(l, 0, 0);
    conv_layer_params* p = l->params;

    if (p->output_size.width != 2 || p->output_size.height != 2 || p->output_size.depth != 1) {
        printf("Wrong output size\n");
        return;
    }

    p->kernels[0] = 1;
    p->kernels[1] = 2;
    p->kernels[2] = -1;
    p->kernels[3] = 0;

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

    is = (size3D) {4, 4, 3};
    l = cnstr_conv_layer(is, ks,2, 2, 1);
    p = l->params;

    if (p->output_size.width != 3 || p->output_size.height != 3 || p->output_size.depth != 2) {
        printf("Wrong output size\n");
        return;
    }

    l->vtable->free(l);

    printf("conv layer forward test OK!\n");
}

void unit_test() {
    pooling_layer_test();
    optimizer_test(1, 0);
    bit_add_learn_test(0);
    build_test();
    mt_dense_test(1);
    dense_test();
    predict_test();
    conv_layer_forward_test();
}
