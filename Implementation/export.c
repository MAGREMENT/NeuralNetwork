#include "export.h"

#include <stdio.h>
#include <stdlib.h>

#include "functions.h"
#include "repository.h"
#include "utils.h"

#define LOG false

inline old_nn* Create(int count, int numbers[]) {
    auto network = alloc_network(count, numbers);
    apply_default_hyper_params(network);
    return network;
}

inline void Dispose(old_nn* ptr) {
    free_network(ptr);
}

inline int GetCount(const old_nn* ptr) {
    return ptr->count;
}

inline int GetOutCount(old_nn* ptr, int layer) {
    return ptr->layers[layer].out_count;
}

inline int GetInCount(old_nn* ptr, int layer) {
    return ptr->layers[layer].in_count;
}

inline void SetWeight(old_nn* ptr, const int layer, const int input, const int output, const double value) {
    ptr->layers[layer].weights[ptr->layers[layer].out_count * input + output] = value;
}

inline double GetWeight(old_nn* ptr, int layer, int input, int output) {
    return ptr->layers[layer].weights[ptr->layers[layer].out_count * input + output];
}

inline void SetBias(old_nn* ptr, int layer, int output, double value) {
    ptr->layers[layer].biases[output] = value;
}

inline double GetBias(old_nn* ptr, int layer, int output) {
    return ptr->layers[layer].biases[output];
}

inline void SetAllWeightsAndBiases(old_nn* ptr, double weights, double biases) {
    set_all_weights_and_biases(ptr, weights, biases);
}

inline double GetLearningRate(old_nn* n) {
    return n->learningRate;
}

inline void SetLearningRate(old_nn* n, double lr) {
    n->learningRate = lr;
}

inline int GetShuffleDataOnIteration(old_nn* n) {
    return n->shuffleDataOnIteration;
}

inline void SetShuffleDataOnIteration(old_nn* n, int sdoi) {
    n->shuffleDataOnIteration = sdoi;
}

inline int GetThreadCount(old_nn* n) {
    return n->threadCount;
}

inline void SetThreadCount(old_nn* n, int th) {
    n->threadCount = th;
}

inline void SetActivationType(old_nn* n, int type, int outputType) {
    set_activation_type(n, type, outputType);
}

inline void SetCostType(old_nn* n, int type) {
    set_cost_type(n, type);
}

inline void SetOptimizerGradientDescent(old_nn* n) {
    set_optimizer(n, create_gradient_descent_optimizer());
}

inline void SetOptimizerMomentum(old_nn* n, double momentum) {
    set_optimizer(n, create_momentum_gradient_descent_optimizer(momentum));
}

inline void SetOptimizerNesterov(old_nn* n, double decay) {
    set_optimizer(n, create_nesterov_optimizer(decay));
}

inline void SetOptimizerAdam(old_nn* n, double delta1, double delta2) {
    set_optimizer(n, create_adam_optimizer(delta1, delta2));
}

inline void SetDataSelectorFullBatch(old_nn* n) {
    set_data_selector(n, create_full_batch_selector());
}

inline void SetDataSelectorMiniBatch(old_nn* n, int batchSize) {
    set_data_selector(n, create_mini_batch_selector(batchSize));
}

inline void SetSchedulerConstant(old_nn* n) {
    set_scheduler(n, constr_constant_scheduler());
}

inline void SetSchedulerIterationDecay(old_nn* n, double proportion) {
    set_scheduler(n, constr_iteration_decay_scheduler(proportion));
}

inline void SetSchedulerExponentialDecay(old_nn* n, double decay) {
    set_scheduler(n, constr_exponential_decay_scheduler(decay));
}

inline void SetSchedulerInverseDecay(old_nn* n, double decay) {
    set_scheduler(n, constr_inverse_decay_scheduler(decay));
}

inline void SetSchedulerCosineDecay(old_nn* n, double endLearningRate, int iterationSpan) {
    set_scheduler(n, constr_cosine_decay_scheduler(endLearningRate, iterationSpan));
}

inline void Predict(old_nn* ptr, double inputs[], int inCount, double outputs[], int outCount) {
    input_data data;
    data.count = inCount;
    data.values = inputs;

    input_data predicted;
    predicted.count = outCount;
    predicted.values = outputs;

    predict(ptr, &data, &predicted);
}

inline old_nn* FromFile(char file[]) {
    auto network = restore(file);
    apply_default_hyper_params(network);
    return network;
}

inline int Save(old_nn* ptr, char file[]) {
    return save(ptr, file);
}

inline void Initialize(old_nn* ptr) {
    init_random();
    old_initialize(ptr);
}

inline void Learn(old_nn* ptr, learning_state* state, double* inputs, int inputCutoff,
        double* expected, int expectedCutoff, int count, int iterations) {
#if LOG
    char* in_seq = alloc_seq_to_str(inputs, inputCutoff * count);
    char* ex_seq = alloc_seq_to_str(expected, expectedCutoff * count);

    flog("Learn called : \n"
        "Inputs : %s\n"
        "InputCutOff : %i\n"
        "Expected : %s\n"
        "ExpectedCutOff : %i\n"
        "Count : %i\n"
        "Iterations : %i\n", in_seq, inputCutoff, ex_seq, expectedCutoff, count, iterations);

    free(in_seq);
    free(ex_seq);
#endif

    test_data* test = alloc_transfer_flattened_data(inputs, inputCutoff, expected, expectedCutoff, count);
    iterative_learn(ptr, test, state, iterations);
    free_transferred_flattened_data(test);
}

inline double Cost(old_nn* ptr, double* inputs, int inputCount, double* expected, int expectedCount) {
    input_data i, e;
    i.count = inputCount;
    i.values = inputs;
    e.count = expectedCount;
    e.values = expected;

    printf("%i", inputCount);
    fflush(stdout);
    return cost(ptr, &i, &e);
}

inline double MultiCost(old_nn* ptr, double* inputs, int inputCutoff, double* expected, int expectedCutoff, int count) {
    test_data* test = alloc_transfer_flattened_data(inputs, inputCutoff, expected, expectedCutoff, count);
    const double result = avg_cost(ptr, test);
    free_transferred_flattened_data(test);
    return result;
}

inline learning_state* CreateState(old_nn* ptr) {
    return alloc_state(ptr);
}

inline void DisposeState(old_nn* ptr, learning_state* state) {
    free_state(ptr, state);
}

void Standardize(double* inputs, int inputCutoff, double* expected, int expectedCutoff, int count) {
    test_data* data = alloc_transfer_flattened_data(inputs, inputCutoff, expected, expectedCutoff, count);
    standardize(data);
    free_transferred_flattened_data(data);
}

void MinMaxScale(double* inputs, int inputCutoff, double* expected, int expectedCutoff, int count) {
    test_data* data = alloc_transfer_flattened_data(inputs, inputCutoff, expected, expectedCutoff, count);
    min_max_scale(data);
    free_transferred_flattened_data(data);
}