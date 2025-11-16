#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "builder.h"
#include "i_o.h"
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

void mnist_run();
void unit_test();

//TODO small test framework

int main(void) {
    //mnist_run();
    unit_test();
    return EXIT_SUCCESS;
}

void mnist_run() {
    //max : 10000
    const int count = 10000;

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

    test_data original = {images, labels, count};
    builder* b = alloc_builder(784);

    b_dense(b, 200);
    b_activation(b, LEAKY_RELU);
    b_dense(b, 100);
    b_activation(b, LEAKY_RELU);
    b_dense(b, 10);
    b_activation(b, SOFTMAX);

    b_opt(b, ADAM, (optimizer_cnstr_args) {.value2 = (double2) {0.9, 0.999}});
    b_sch(b, CONSTANT, (scheduler_cnstr_args) {.value = 0.0});
    b_ds(b, MINI_BATCH, (data_selector_cnstr_args) {.value = 64});

    b->cost_type = BINARY_CROSS_ENTROPY;

    b->shuffleDataOnIteration = 1;
    b->learningRate = 0.01;

    neural_network* n = build_free(b, def_b_params());
    initialize(n);

    shuffle_test_data(original, n, 3);
    test_data training;
    test_data testing;
    separate_test_data(original, 784, 10, &training, &testing, 0.8);

    printf("Iteration 0 : TRAINING => Cost -> %f | Accuracy -> %f TESTING => Cost -> %f | Accuracy -> %f\n",
        get_avg_cost(n, training), get_classification_accuracy(n, training), get_avg_cost(n, testing), get_classification_accuracy(n, testing));
    learning_state* state = alloc_state(n);

    clock_t start = clock();

    for (int i = 0; i < 5; i++) {
        iterative_learn(n, training, state, 1);
        printf("Iteration %d : TRAINING => Cost -> %f | Accuracy -> %f TESTING => Cost -> %f | Accuracy -> %f\n", i + 1,
        get_avg_cost(n, training), get_classification_accuracy(n, training), get_avg_cost(n, testing), get_classification_accuracy(n, testing));
    }

    clock_t end = clock();

    printf("Learning time : %f s", (double)(end - start) / CLOCKS_PER_SEC);

    free_neural_network(n, 1);
    free(images);
    free(labels);
}

void save_test() {
    builder* b = alloc_builder(4);
    b_dense(b, 4);
    b_activation(b, SIGMOID);
    b_dense(b, 5);
    b_activation(b, SOFTMAX);

    b_ds(b, FULL_BATCH, (data_selector_cnstr_args) {.value = 0});
    b_sch(b, CONSTANT, (scheduler_cnstr_args) {.value = 0.0});
    b_opt(b, SIMPLE, (optimizer_cnstr_args) {.value = 0.0});

    b->cost_type = BINARY_CROSS_ENTROPY;
    b->shuffleDataOnIteration = 0;

    neural_network* n = build_free(b, st_b_params());
    initialize(n);

    double in[4 * 16];
    double out[5 * 16];

    generate_binary_inputs(in, 4);
    generate_classify_sum_outputs(out, in, 4);
    test_data test = {in, out, 16};

    iterative_learn_stateless(n, test,5);

    const double c1 = get_avg_cost(n, test);
    const double a1 = get_classification_accuracy(n, test);

    const char file[] = "save_test.nn";
    if (!save_parameters(n, file)) {
        printf("Failed to save test parameters\n");
        goto free;
    }

    set_all_weights_and_biases(n->layers[0], 10, 10);
    set_all_weights_and_biases(n->layers[2], 10, 10);

    if (!restore_parameters(n, file)) {
        printf("Failed to restore parameters\n");
        goto free;
    }

    const double c2 = get_avg_cost(n, test);
    const double a2 = get_classification_accuracy(n, test);

    if (!def_deq(c1, c2)) {
        printf("Wrong cost\n");
        goto free;
    }

    if (!def_deq(a1, a2)) {
        printf("Wrong accuracy\n");
        goto free;
    }

    printf("Save test OK!\n");

    free :

    remove(file);
    free_neural_network(n, 1);
}

void generate_binary_inputs_tests() {
    const double expected[] = {
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        1, 1, 0,
        0, 0, 1,
        1, 0, 1,
        0, 1, 1,
        1, 1, 1};

    double inputs[2 * 2 * 2 * 3];

    generate_binary_inputs(inputs, 3);
    for (int i = 0; i < sizeof(inputs) / sizeof(double); i++) {
        if (!def_deq(expected[i], inputs[i])) {
            printf("Wrong value\n");
            return;
        }
    }

    const double o1Expected[] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };

    double o1[2 * 2 * 2 * 4];

    generate_classify_sum_outputs(o1, inputs, 3);
    for (int i = 0; i < sizeof(o1) / sizeof(double); i++) {
        if (!def_deq(o1[i], o1Expected[i])) {
            printf("Wrong value\n");
            return;
        }
    }

    const double o2Expected[] = {
        0, 0,
        1, 0,
        1, 0,
        0, 1,
        1, 0,
        0, 1,
        0, 1,
        1, 1
    };

    double o2[2 * 2 * 2 * 2];
    generate_binary_sum_outputs(o2, inputs, 3);
    for (int i = 0; i < sizeof(o2) / sizeof(double); i++) {
        if (!def_deq(o2[i], o2Expected[i])) {
            printf("Wrong value\n");
            return;
        }
    }

    printf("generate binary inputs test OK!\n");
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
            goto free;
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
            goto free;
        }
    }

    //TODO backward & i3 & avg

    printf("pooling layer test OK!\n");

    free :

    l->vtable->free(l);
}

typedef struct opt_builder {
    int opt;
    optimizer_cnstr_args opt_args;
    double learning_rate;
} opt_builder;

static void get_simple_cost_arr(neural_network* dummy, const double* in, double* predicted, const double* expected, double* cost) {
    predict(dummy, in, predicted);
    for (int i = 0; i < 4; i++) {
        cost[i] = predicted[i] - expected[i];
    }
}

static void print_costs(double* costs, const int count) {
    if (count == 0) return;

    printf("%.4f", costs[0]);

    for (int i = 1; i < count; i++) {
        printf(" | %.4f", costs[i]);
    }

    printf("\n");
}

void optimizer_test(const int error, const int verbose) {
    const opt_builder builds[] = {
        {SIMPLE, (optimizer_cnstr_args) {.value = 0}, 0.1},
        {FREE_MOMENTUM, (optimizer_cnstr_args) {.value = 0.2}, 0.1},
        {PROPORTIONAL_MOMENTUM, (optimizer_cnstr_args) {.value = 0.9}, 0.1},
        {SIMPLIFIED_NESTEROV, (optimizer_cnstr_args) {.value = 0.3}, 0.1},
        {RMS_PROP, (optimizer_cnstr_args) {.value = 0.9}, 0.1},
        {ADAM, (optimizer_cnstr_args) {.value2 = (double2) {0.9, 0.999}}, 0.1},
        {ADAGRAD, (optimizer_cnstr_args) {.value = 0.0}, 0.1},
        {ADADELTA, (optimizer_cnstr_args) {.value = 0.9}, 0.1},
        {ADAMAX, (optimizer_cnstr_args) {.value2 = (double2) {0.9, 0.999}}, 0.1},
        {ADAMW, (optimizer_cnstr_args) {.value3 = (double3) {0.9, 0.999, 0.01}}, 0.1},
    };

    neural_network* dummy = alloc_neural_network(1);
    layer* l = cnstr_dense_layer(1, 4, initialize_dense_to_one);
    dummy->layers[0] = l;
    initialize(dummy);

    dense_layer_params* p = l->data;

    double in[] = {1};
    double expected[] = {-7, 15, 0, 3};
    double predicted[4];
    double cost[4];

    for (int i = 0; i < sizeof(builds) / sizeof(opt_builder); ++i) {
        if (verbose) printf("%s : \n", opt_names[builds[i].opt]);

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
                        printf("%s : Fail\n", opt_names[builds[i].opt]);
                        opt->vtable->free_state(state, dummy);
                        opt->vtable->free(opt);
                        goto free;
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

    printf("optimizer test OK!\n");

    free :

    free_neural_network(dummy, 1);
}

int a(builder* b, const opt_builder builds[], const int buildCount, const test_data test, const int verbose, double (*get_acc)(const neural_network*, test_data)) {
    for (int i = 0; i < buildCount; i++) {
        b_opt(b, builds[i].opt, builds[i].opt_args);
        b->learningRate = builds[i].learning_rate;

        neural_network* n = build(b, st_b_params());
        initialize(n);

        learning_state* state = alloc_state(n);

        double cost = get_avg_cost(n, test);
        int fail = 0;

        for (int j = 0; j < 10; j++) {
            iterative_learn(n, test, state, 100);

            const double buffer = get_avg_cost(n, test);
            if (buffer >= cost) {
                if (verbose) fail = 1;
                else {
                    printf("bit learn cost fail for %s and iteration %d\n", opt_names[builds[i].opt], j * 10);
                    free_state(n, state);
                    free_neural_network(n, 1);
                    return 1;
                }
            }

            cost = buffer;
        }

        if (verbose) {
            printf("%s %s-> Cost : %f | Accuracy = %f\n", opt_names[builds[i].opt], fail ? "(FAIL) " : "", cost, get_acc(n, test));
        }

        free_state(n, state);
        free_neural_network(n, 1);
    }

    return 0;
}

void binary_sum_learn_test(const int verbose) {
    const opt_builder builds[] = {
        {SIMPLE, (optimizer_cnstr_args) {.value = 0}, 1},
        {FREE_MOMENTUM, (optimizer_cnstr_args) {.value = 0.1}, 1},
        {PROPORTIONAL_MOMENTUM, (optimizer_cnstr_args) {.value = 0.9}, 1},
        {SIMPLIFIED_NESTEROV, (optimizer_cnstr_args) {.value = 0.8}, 1},
        {RMS_PROP, (optimizer_cnstr_args) {.value = 0.9}, 0.1},
        {ADAM, (optimizer_cnstr_args) {.value2 = (double2) {0.9, 0.999}}, 1},
        {ADAGRAD, (optimizer_cnstr_args) {.value = 0.0}, 1},
        {ADADELTA, (optimizer_cnstr_args) {.value = 0.9}, 1},
        {ADAMAX, (optimizer_cnstr_args) {.value2 = (double2) {0.9, 0.999}}, 1},
        {ADAMW, (optimizer_cnstr_args) {.value3 = (double3) {0.9, 0.999, 0.01}}, 1}
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

    double in[128 * 7];
    double exp[128 * 3];

    generate_binary_inputs(in, 7);
    generate_binary_sum_outputs(exp, in, 7);

    test_data test;
    test.count = 128;
    test.inputs = in;
    test.expected = exp;

    if (verbose) printf("\n");
    if (a(b, builds, sizeof(builds) / sizeof(opt_builder), test, verbose, get_binary_accuracy)) goto free;

    free_builder(b);
    b = alloc_builder(7);
    b_dense(b, 4);
    b_activation(b, LEAKY_RELU);
    b_dense(b, 8);
    b_activation(b, SOFTMAX);

    b_ds(b, FULL_BATCH, (data_selector_cnstr_args) {.value = 0});
    b_sch(b, CONSTANT, (scheduler_cnstr_args) {.value = 0.0});

    b->cost_type = BINARY_CROSS_ENTROPY;
    b->shuffleDataOnIteration = 0;

    double exp2[128 * 8];

    generate_classify_sum_outputs(exp2, in, 7);

    test.count = 128;
    test.inputs = in;
    test.expected = exp2;

    if (verbose) printf("\n---------\n\n");

    if (a(b, builds, sizeof(builds) / sizeof(opt_builder), test, verbose, get_classification_accuracy)) goto free;

    if (verbose) printf("\n");
    printf("bit add learn test OK!\n");

    free :

    free_builder(b);
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

    neural_network* n = build_free(b, st_b_params());

    if (n->layerCount != 5) {
        printf("Wrong layer count\n");
        goto free;
    }

    if (n->layers[0]->in_count != 9 || n->layers[0]->out_count != 7 ||
        n->layers[1]->in_count != 7 || n->layers[1]->out_count != 7 ||
        n->layers[2]->in_count != 7 || n->layers[2]->out_count != 7 ||
        n->layers[3]->in_count != 7 || n->layers[3]->out_count != 3 ||
        n->layers[4]->in_count != 3 || n->layers[4]->out_count != 3) {
        printf("Wrong layer i/o\n");
        goto free;
    }

    if (n->layers[0]->parameters_count != 9 * 7 + 7 ||
        n->layers[1]->parameters_count != 0 ||
        n->layers[2]->parameters_count != 0 ||
        n->layers[3]->parameters_count != 7 * 3 + 3 ||
        n->layers[4]->parameters_count != 0) {
        printf("Wrong layer gradient count\n");
        goto free;
    }

    if (n->layers[0]->initialize != initialize_dense_he
        || n->layers[3]->initialize != initialize_dense_random) {
        printf("Wrong dense layer initialize func\n");
        goto free;
    }

    //TODO more tests
    printf("builder test OK!\n");

    free :

    free_neural_network(n, 1);
}

void mt_dense_test(const int count, const int verbose) {
    const int inCount = count;
    const int outCount = count;

    double* in = malloc(inCount * sizeof(double));
    double* out1 = malloc(outCount * sizeof(double));
    double* out2 = malloc(outCount * sizeof(double));
#ifdef _MSC_VER
    double* out3 = malloc(outCount * sizeof(double));
#endif

    double* grads1 = malloc((inCount * outCount + outCount) * sizeof(double));
    double* grads2 = malloc((inCount * outCount + outCount) * sizeof(double));

    for (int i = 0; i < inCount; i++) {
        in[i] = rand_d(-5, 5);
    }

    layer* single = cnstr_dense_layer(inCount, outCount, initialize_dense_to_zero);
    layer* multi = cnstr_multi_thread_dense_layer(inCount, outCount, 8, initialize_dense_to_zero);

    dense_layer_params* sp = single->data;
    dense_layer_params* mp = multi->data;

    clock_t singleTime = 0;
    clock_t multiTime = 0;

#ifdef _MSC_VER
    layer* cuda = cnstr_cuda_dense_layer(inCount, outCount, 256, initialize_dense_to_zero);
    dense_layer_params* cp = cuda->data;
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
                printf("Not same forward value\n");
                goto free;
            }
        }

#ifdef _MSC_VER
        for (int i = 0; i < outCount; i++) {
            if (!def_deq(out1[i], out3[i])) {
                printf("Not same forward value\n");
                goto free;
            }
        }
#endif

        single->vtable->backward(single, NULL, in, out1);

        multi->vtable->backward(multi, NULL, in, out2);

        for (int i = 0; i < inCount; i++) {
            if (!def_deq(out1[i], out2[i])) {
                printf("Not same backward value\n");
                goto free;
            }
        }

        single->vtable->deltas_to_gradients(single, in, out1, grads1);

        multi->vtable->deltas_to_gradients(multi, in, out1, grads2);

        for (int i = 0; i < inCount * outCount + outCount; i++) {
            if (!def_deq(grads1[i], grads2[i])) {
                printf("Not same dtg value\n");
                goto free;
            }
        }
    }

    if (verbose) {
        printf("Single thread time : %f s\n", (double)singleTime / CLOCKS_PER_SEC);
        printf("Multi thread time : %f s\n", (double)multiTime / CLOCKS_PER_SEC);
#ifdef _MSC_VER
        printf("GPU time : %f s\n", (double)cudaTime / CLOCKS_PER_SEC);
#endif
    }

    printf("multi-thread dense layer test OK!\n");

    free:

    free(single);
    free(multi);
#ifdef _MSC_VER
    free(cuda);
    free(out3);
#endif
    free(in);
    free(out1);
    free(out2);
    free(grads1);
    free(grads2);
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
        goto free;
    }
    if (!def_deq(o[1], 4)) {
        printf("DENSE TEST FAIL !\n");
        goto free;
    }
    if (!def_deq(o[2], 5)) {
        printf("DENSE TEST FAIL !\n");
        goto free;
    }

    double o2[2];

    n->layers[1]->vtable->forward(n->layers[1], o, o2);

    if (!def_deq(o2[0], 4)) {
        printf("DENSE TEST FAIL !\n");
        goto free;
    }
    if (!def_deq(o2[1], 10)) {
        printf("DENSE TEST FAIL !\n");
        goto free;
    }

    double o3[2];

    predict(n, i, o3);

    if (!def_deq(o2[0], o3[0])) {
        printf("DENSE TEST FAIL !\n");
        goto free;
    }
    if (!def_deq(o2[1], o3[1])) {
        printf("DENSE TEST FAIL !\n");
        goto free;
    }

    n->cost_vtable = cost_vtables + MEAN_SQUARE;
    double e[] = {2, 7};

    double cost = get_cost(n, i, e);

    if (!def_deq(cost, 4 + 9)) {
        printf("DENSE TEST FAIL !\n");
        goto free;
    }

    n->optimizer = cnstr_simple_optimizer(); //TODO to set_optimizer
    const double learningRate = 0.01;

    for (int epoch = 0; epoch < 10; epoch++) {

        learn_stateless(n, (test_data){i, e, 1}, (range){1, 0, 1}, learningRate);
        const double cost2 = get_cost(n, i, e);

        if (cost2 >= cost) {
            printf("Dense test cost lowering fail\n");
            goto free;
        }

        cost = cost2;
    }

    printf("dense test OK!\n");

    free:

    free_neural_network(n, 1);
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
            goto free;
        }
    }

    printf("predict test OK!\n");

    free :

    free_neural_network(n, 1);
}

void conv_layer_forward_test() {
    size3D is = {3, 3, 1};
    size2D ks = {2, 2};
    layer* l = cnstr_conv_layer(is, ks,1, 1, 0, initialize_conv_random);

    set_kernels_and_biases(l, 0, 0);
    conv_layer_params* p = l->data;

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
            goto free;
        }
    }

    l->vtable->free(l);

    is = (size3D) {4, 4, 3};
    l = cnstr_conv_layer(is, ks,2, 2, 1, initialize_conv_random);
    p = l->data;

    if (p->output_size.width != 3 || p->output_size.height != 3 || p->output_size.depth != 2) {
        printf("Wrong output size\n");
        goto free;
    }

    printf("conv layer forward test OK!\n");

    free:

    l->vtable->free(l);
}

void unit_test() {
    save_test();
    generate_binary_inputs_tests();
    pooling_layer_test();
    optimizer_test(1, 0);
    binary_sum_learn_test(0);
    build_test();
    mt_dense_test(1000, 0);
    dense_test();
    predict_test();
    conv_layer_forward_test();
}
