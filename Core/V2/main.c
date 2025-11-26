
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#include "builder.h"
#include "i_o.h"
#include "multi-threading.h"
#include "Layers/Types/convolutional_layer.h"
#include "neural_network.h"
#include "tester.h"
#include "Layers/Types/activation_layer.h"
#include "Layers/Types/dense_layer.h"
#include "Layers/Types/pooling_layer.h"
#include "Optimizers/Types/simple_optimizer.h"
#include "Util/double_util.h"
#include "Util/math_util.h"
#include "Util/rand_util.h"
#include "Util/Collections/queue.h"

#ifdef _MSC_VER
#include "Layers/Types/Cuda/cuda_dense_layer.cuh"
#endif

void mnist_run();
void unit_test();
void unit_test2();

int main(void) {
    //mnist_run();
    unit_test(); //TODO convert
    unit_test2();
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
    builder* b = alloc_builder_3D((size3D){28, 28, 1});

    b_dense(b, 200);
    b_activation(b, LEAKY_RELU);
    b_dense(b, 100);
    b_activation(b, LEAKY_RELU);
    b_dense(b, 10);
    b_activation(b, SOFTMAX);

    /*b_conv(b, (size2D){3, 3}, 8, 1, 1);
    b_activation(b, LEAKY_RELU);
    b_pooling(b, POOLING_MAX, (size2D){2, 2}, 1, 0);
    b_conv(b, (size2D){3, 3}, 16, 1, 1);
    b_activation(b, LEAKY_RELU);
    b_pooling(b, POOLING_MAX, (size2D){2, 2}, 1, 0);
    b_dense(b, 128);
    b_activation(b, LEAKY_RELU);
    b_dense(b, 10);
    b_activation(b, SOFTMAX);*/

    b_opt(b, ADAM, TCA_DOUBLE2(0.9, 0.999));
    b_sch(b, CONSTANT, TCA_NONE);
    b_ds(b, MINI_BATCH, TCA_INT(64));

    b->cost_type = BINARY_CROSS_ENTROPY;

    b->shuffleDataOnIteration = 1;
    b->learningRate = 0.01;

    builder_params bp = def_b_params();
    //bp.dense_mt_threshold = INT_MAX;
    bp.batch_threads = 1;
    neural_network* n = build_free(b, bp);
    initialize(n);

    shuffle_test_data(original, n, 3);
    test_data training;
    test_data testing;
    separate_test_data(original, 784, 10, &training, &testing, 0.8);

    printf("Iteration 0 : TRAINING => Cost -> %f | Accuracy -> %f TESTING => Cost -> %f | Accuracy -> %f\n",
        get_avg_cost(n, training), get_classification_accuracy(n, training), get_avg_cost(n, testing), get_classification_accuracy(n, testing));
    learning_data* data = alloc_learning_data(n);

    clock_t start = clock();

    for (int i = 0; i < 10; i++) {
        iterative_learn(n, training, data, 1);
        printf("Iteration %d : TRAINING => Cost -> %f | Accuracy -> %f TESTING => Cost -> %f | Accuracy -> %f\n", i + 1,
        get_avg_cost(n, training), get_classification_accuracy(n, training), get_avg_cost(n, testing), get_classification_accuracy(n, testing));
    }

    clock_t end = clock();

    printf("Learning time : %f s", (double)(end - start) / CLOCKS_PER_SEC);

    free_learning_data(n, data);
    free_neural_network(n, 1);
    free(images);
    free(labels);
}

TEST(builder_yaml_test) {
    builder* b = alloc_builder_3D((size3D){28, 28, 1});

    b_dense(b, 200);
    b_activation(b, LEAKY_RELU);
    b_dense(b, 100);
    b_activation(b, LEAKY_RELU);
    b_dense(b, 10);
    b_activation(b, SOFTMAX);

    b_opt(b, ADAM, TCA_DOUBLE2(0.9, 0.999));
    b_sch(b, CONSTANT, TCA_NONE);
    b_ds(b, MINI_BATCH, TCA_INT(64));

    b->cost_type = BINARY_CROSS_ENTROPY;

    b->shuffleDataOnIteration = 1;
    b->learningRate = 0.01;

    yaml_writer* w = alloc_yaml_writer();
    to_yaml(b, w);
    save_yaml(w, "yaml-test.yaml");

    yaml_reader* r = alloc_yaml_reader();
    download_yaml(r, "yaml-test.yaml");

    ASSERT(w->lines->count == r->lines->count);

    for (int i = 0; i < w->lines->count; i++) {
        const yaml_line wl = l_get(w->lines, yaml_line, i);
        const yaml_line rl = l_get(r->lines, yaml_line, i);

        ASSERT(wl.indentation == rl.indentation);
        ASSERT(wl.is_array == rl.is_array);
        ASSERT(strcmp(wl.name, rl.name) == 0);
        ASSERT(strcmp(wl.value, rl.value) == 0);
    }

    TEARDOWN
    free_builder(b);
    free_yaml_writer(w);
    free_yaml_reader(r);
}

TEST(queue_test) {
    queue* q = alloc_queue(3, sizeof(int));

    ASSERT_M(is_empty(q), "Queue should be empty")
    ASSERT_M(!is_full(q), "Queue should not be full")

    q_enq(q, int, 1);

    ASSERT_M(!is_empty(q), "Queue should not be empty")
    ASSERT_M(!is_full(q), "Queue should not be full")

    q_enq(q, int, 2);
    int buffer = q_deq(q, int);

    ASSERT_M(buffer == 1, "Buffer should be 1")
    ASSERT_M(!is_empty(q), "Queue should not be empty")
    ASSERT_M(!is_full(q), "Queue should not be full")

    q_enq(q, int, 3);

    ASSERT_M(!is_empty(q), "Queue should not be empty")
    ASSERT_M(!is_full(q), "Queue should not be full")

    q_enq(q, int, 4);

    ASSERT_M(!is_empty(q), "Queue should not be empty")
    ASSERT_M(is_full(q), "Queue should be full")

    q_enq(q, int, 5);

    ASSERT_M(!is_empty(q), "Queue should not be empty")
    ASSERT_M(!is_full(q), "Queue should not be full")

    buffer = q_deq(q, int);

    ASSERT_M(buffer == 2, "Buffer should be 2")

    buffer = q_deq(q, int);

    ASSERT_M(buffer == 3, "Buffer should be 3")

    buffer = q_deq(q, int);

    ASSERT_M(buffer == 4, "Buffer should be 4")

    buffer = q_deq(q, int);

    ASSERT_M(buffer == 5, "Buffer should be 5")

    ASSERT_M(is_empty(q), "Queue should be empty");

    const int count = 10000;
    for (int i = 0; i < count; i++) {
        q_enq(q, int, i);
    }

    for (int i = 0; i < count; i++) {
        const int curr = q_deq(q, int);
        ASSERT(curr == i);
    }

    ASSERT_M(is_empty(q), "Queue should be empty");

    TEARDOWN
    free_queue(q);
}

TEST(save_test) {
    builder* b = alloc_builder(4);
    b_dense(b, 4);
    b_activation(b, SIGMOID);
    b_dense(b, 5);
    b_activation(b, SOFTMAX);

    b_ds(b, FULL_BATCH, TCA_NONE);
    b_sch(b, CONSTANT, TCA_NONE);
    b_opt(b, SIMPLE, TCA_NONE);

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
    ASSERT_M(save_parameters(n, file), "Failed to save test parameters")

    set_all_weights_and_biases(n->layers[0], 10, 10);
    set_all_weights_and_biases(n->layers[2], 10, 10);

    ASSERT_M(restore_parameters(n, file), "Failed to restore parameters")

    const double c2 = get_avg_cost(n, test);
    const double a2 = get_classification_accuracy(n, test);

    ASSERT_M(def_deq(c1, c2), "Wrong cost")
    ASSERT_M(def_deq(a1, a2), "Wrong accuracy")

    TEARDOWN
    remove(file);
    free_neural_network(n, 1);
}

typedef struct arr_tp_test{
    const int* arr;
    int* result;
    int n;
} arr_tp_test;

unsigned long copy_arr_tp_test(void* params) {
    arr_tp_test* arr = params;
    arr->result[arr->n] = arr->arr[arr->n];
    return 0;
}

TEST(threadpool_test) {
    const int count = 1000;

    thread_pool* tp = alloc_thread_pool(4);
    job_group* group = alloc_job_group(count);

    int* n = malloc(sizeof(int) * count);
    int* r = malloc(sizeof(int) * count);
    arr_tp_test* arrs = malloc(sizeof(arr_tp_test) * count);

    for (int i = 0; i < count; i++) {
        n[i] = i + 1;
        arrs[i] = (arr_tp_test) {n, r, i};
    }

    for (int i = 0; i < count; i++) {
        add_job(tp, copy_arr_tp_test, arrs + i, group);
    }

    wait_for_group(group);

    for (int i = 0; i < count; i++) {
        ASSERT_M(n[i] == r[i], "Wrong result");
    }

    TEARDOWN
    free(n);
    free(r);
    free(arrs);
    free_thread_pool(tp);
    free_job_group(group);
}

void unit_test2() {
    CONTEXT
    ADD_TEST(queue_test);
    ADD_TEST(save_test);
    ADD_TEST(threadpool_test);
    ADD_TEST(builder_yaml_test);
    RUN
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
    tc_cnstr_args opt_args;
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
        {SIMPLE, TCA_NONE, 0.1},
        {FREE_MOMENTUM, TCA_DOUBLE(0.2), 0.1},
        {PROPORTIONAL_MOMENTUM, TCA_DOUBLE(0.9), 0.1},
        {SIMPLIFIED_NESTEROV, TCA_DOUBLE(0.3), 0.1},
        {RMS_PROP, TCA_DOUBLE(0.9), 0.1},
        {ADAM, TCA_DOUBLE2(0.9, 0.999), 0.1},
        {ADAGRAD, TCA_NONE, 0.1},
        {ADADELTA, TCA_DOUBLE(0.9), 0.1},
        {ADAMAX, TCA_DOUBLE2(0.9, 0.999), 0.1},
        {ADAMW, TCA_DOUBLE3(0.9, 0.999, 0.01), 0.1}
    };

    neural_network* dummy = alloc_neural_network(1);
    layer* l = cnstr_dense_layer(1, 4, initialize_dense_to_one);
    dummy->layers[0] = l;
    initialize(dummy);

    double in[] = {1};
    double expected[] = {-7, 15, 0, 3};
    double predicted[4];
    double cost[4];

    for (int i = 0; i < sizeof(builds) / sizeof(opt_builder); ++i) {
        if (verbose) printf("%s : \n", opt_metadata[builds[i].opt].name);

        l->parameters[0] = 1;
        l->parameters[1] = 1;
        l->parameters[2] = 1;
        l->parameters[3] = 1;

        optimizer* opt = cnstr_optimizer(builds[i].opt, builds[i].opt_args);
        void* state = opt->vtable->cnstr_state(opt, dummy);
        optimizer_args args = {builds[i].learning_rate, 0, 0, state};

        get_simple_cost_arr(dummy, in, predicted, expected, cost);
        if (verbose) print_costs(cost, 4);

        for (int iteration = 0; iteration < 10; iteration++) {
            args.iteration += 1;
            opt->vtable->apply_gradients(opt, l->parameters, cost, 4, args);

            if (verbose) printf("Iteration %d : \n", iteration + 1);

            double buffer[4];
            get_simple_cost_arr(dummy, in, predicted, expected, buffer);

            if (error) {
                for (int j = 0; j < 4; j++) {
                    if (fabs(buffer[j]) > fabs(cost[j])) {
                        printf("%s : Fail\n", opt_metadata[builds[i].opt].name);
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

        learning_data* data = alloc_learning_data(n);

        double cost = get_avg_cost(n, test);
        int fail = 0;

        for (int j = 0; j < 10; j++) {
            iterative_learn(n, test, data, 100);

            const double buffer = get_avg_cost(n, test);
            if (buffer >= cost) {
                if (verbose) fail = 1;
                else {
                    printf("bit learn cost fail for %s and iteration %d\n", opt_metadata[builds[i].opt].name, j * 10);
                    free_learning_data(n, data);
                    free_neural_network(n, 1);
                    return 1;
                }
            }

            cost = buffer;
        }

        if (verbose) {
            printf("%s %s-> Cost : %f | Accuracy = %f\n", opt_metadata[builds[i].opt].name, fail ? "(FAIL) " : "", cost, get_acc(n, test));
        }

        free_learning_data(n, data);
        free_neural_network(n, 1);
    }

    return 0;
}

void binary_sum_learn_test(const int verbose) {
    const opt_builder builds[] = {
        {SIMPLE, TCA_NONE, 1},
        {FREE_MOMENTUM, TCA_DOUBLE(0.2), 1},
        {PROPORTIONAL_MOMENTUM, TCA_DOUBLE(0.9), 1},
        {SIMPLIFIED_NESTEROV, TCA_DOUBLE(0.3), 1},
        {RMS_PROP, TCA_DOUBLE(0.9), 1},
        {ADAM, TCA_DOUBLE2(0.9, 0.999), 1},
        {ADAGRAD, TCA_NONE, 1},
        {ADADELTA, TCA_DOUBLE(0.9), 1},
        {ADAMAX, TCA_DOUBLE2(0.9, 0.999), 1},
        {ADAMW, TCA_DOUBLE3(0.9, 0.999, 0.01), 1}
    };

    builder* b = alloc_builder(7);
    b_dense(b, 4);
    b_activation(b, SIGMOID);
    b_dense(b, 3);
    b_activation(b, SIGMOID);

    b_ds(b, FULL_BATCH, TCA_NONE);
    b_sch(b, CONSTANT, TCA_NONE);

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

    b_ds(b, FULL_BATCH, TCA_NONE);
    b_sch(b, CONSTANT, TCA_NONE);

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

    b_opt(b, SIMPLIFIED_NESTEROV, TCA_DOUBLE(0.9));
    b_ds(b, FULL_BATCH, TCA_NONE);
    b_sch(b, COSINE_DECAY, TCA_DOUBLE_INT(0.5, 20));

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

    thread_pool* pool = alloc_thread_pool(8);
    layer* layers[] = {
        cnstr_dense_layer(inCount, outCount, initialize_dense_to_zero),
        cnstr_multi_thread_dense_layer(inCount, outCount, pool, 8, initialize_dense_to_zero),
#ifdef _MSC_VER
        cnstr_cuda_dense_layer(inCount, outCount, 256, initialize_dense_to_zero)
#endif
    };

    const int layerCount = sizeof(layers) / sizeof(layer*);
    const int operations = 3;

    double* in = malloc(inCount * sizeof(double));
    double** outs = malloc(layerCount * sizeof(double*));
    double** grads = malloc(layerCount * sizeof(double*));
    clock_t* times = malloc(layerCount * operations * sizeof(clock_t));

    for (int i = 0; i < layerCount; i++) {
        outs[i] = malloc(outCount * sizeof(double));
        grads[i] = malloc(layers[i]->parameters_count * sizeof(double));
        for (int j = 0; j < operations; j++) {
            times[i * operations + j] = 0;
        }
    }

    for (int i = 0; i < inCount; i++) {
        in[i] = rand_d(-5, 5);
    }

    clock_t s, e;

    for (int iteration = 0; iteration < 5; iteration++) {
        for (int i = 0; i < inCount * outCount; i++) {
            const double d = rand_d(-5, 5);
            for (int j = 0; j < layerCount; j++) {
                layers[j]->parameters[i] = d;
            }
        }

        for (int i = 0; i < outCount; i++) {
            const double d = rand_d(-5, 5);
            for (int j = 0; j < layerCount; j++) {
                layers[j]->parameters[inCount * outCount + i] = d;
            }
        }

#ifdef _MSC_VER
        on_parameters_change(layers[layerCount - 1]);
#endif

        int op = 0;

        for (int j = 0; j < layerCount; j++) {
            const layer* l = layers[j];
            s = clock();
            l->vtable->forward(l, in, outs[j]);
            e = clock();
            times[j * operations + op] = e - s;
        }

        for (int j = 1; j < layerCount; j++) {
            for (int i = 0; i < outCount; i++) {
                if (!def_deq(outs[0][i], outs[j][i])) {
                    printf("Not same forward value\n");
                    goto free;
                }
            }
        }

        op++;

        for (int j = 0; j < layerCount; j++) {
            const layer* l = layers[j];
            s = clock();
            l->vtable->backward(l, NULL, in, outs[j]);
            e = clock();
            times[j * operations + op] = e - s;
        }

        for (int j = 1; j < layerCount; j++) {
            for (int i = 0; i < outCount; i++) {
                if (!def_deq(outs[0][i], outs[j][i])) {
                    printf("Not same backward value\n");
                    goto free;
                }
            }
        }

        op++;

        for (int j = 0; j < layerCount; j++) {
            const layer* l = layers[j];
            s = clock();
            l->vtable->deltas_to_gradients(l, in, outs[j], grads[j]);
            e = clock();
            times[j * operations + op] = e - s;
        }

        for (int j = 1; j < layerCount; j++) {
            for (int i = 0; i < outCount; i++) {
                if (!def_deq(grads[0][i], grads[j][i])) {
                    printf("Not same dtg value\n");
                    goto free;
                }
            }
        }
    }

    if (verbose) {
        const char* opNames[] = {"Forward", "Backward", "DTG"};
        const char* layerNames[] = {"Single-Thread", "Multi-Thread", "GPU"};
        for (int op = 0; op < operations; op++) {
            printf("%s\n", opNames[op]);
            for (int j = 0; j < layerCount; j++) {
                printf("%s time : %f s\n", layerNames[j], (double)times[j * operations + op] / CLOCKS_PER_SEC);
            }
            printf("\n\n");
        }
    }

    printf("multi-thread dense layer test OK!\n");

    free:


    free_thread_pool(pool);
    free(in);
    for (int j = 0; j < layerCount; j++) {
        layers[j]->vtable->free(layers[j]);
        free(outs[j]);
        free(grads[j]);
    }

    free(times);
    free(outs);
    free(grads);
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

        learn_stateless(n, (test_data){i, e, 1}, (iteration_range){1, 0, 1}, learningRate);
        const double cost2 = get_cost(n, i, e);

        if (cost2 >= cost) {
            printf("Dense test cost lowering fail\n"); //TODO fix
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

    l->parameters[0] = 1;
    l->parameters[1] = 2;
    l->parameters[2] = -1;
    l->parameters[3] = 0;

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
    generate_binary_inputs_tests();
    pooling_layer_test();
    optimizer_test(1, 0);
    binary_sum_learn_test(0);
    build_test();
    mt_dense_test(1000, 1);
    dense_test();
    predict_test();
    conv_layer_forward_test();
}