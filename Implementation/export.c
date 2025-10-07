#include "export.h"

#include <stdio.h>
#include <stdlib.h>

#include "repository.h"
#include "utils.h"

#define LOG true

inline neural_network* Create(int count, int numbers[]) {
    auto network = alloc_network(count, numbers);
    apply_default_hyper_params(network);
    return network;
}

inline void Dispose(neural_network* ptr) {
    free_network(ptr);
}

inline int GetCount(const neural_network* ptr) {
    return ptr->count;
}

inline int GetOutCount(neural_network* ptr, int layer) {
    return ptr->layers[layer].out_count;
}

inline int GetInCount(neural_network* ptr, int layer) {
    return ptr->layers[layer].in_count;
}

inline void SetWeight(neural_network* ptr, const int layer, const int input, const int output, const double value) {
    ptr->layers[layer].weights[ptr->layers[layer].out_count * input + output] = value;
}

inline double GetWeight(neural_network* ptr, int layer, int input, int output) {
    return ptr->layers[layer].weights[ptr->layers[layer].out_count * input + output];
}

inline void SetBias(neural_network* ptr, int layer, int output, double value) {
    ptr->layers[layer].biases[output] = value;
}

inline double GetBias(neural_network* ptr, int layer, int output) {
    return ptr->layers[layer].biases[output];
}

inline void SetAllWeightsAndBiases(neural_network* ptr, double weights, double biases) {
    set_all_weights_and_biases(ptr, weights, biases);
}

inline double GetLearningRate(neural_network* n) {
    return n->learningRate;
}

inline void SetLearningRate(neural_network* n, double lr) {
    n->learningRate = lr;
}

inline int GetShuffleDataOnIteration(neural_network* n) {
    return n->shuffleDataOnIteration;
}

inline void SetShuffleDataOnIteration(neural_network* n, int sdoi) {
    n->shuffleDataOnIteration = sdoi;
}

inline void SetActivationType(neural_network* n, int type, int outputType) {
    set_activation_type(n, type, outputType);
}

inline void SetCostType(neural_network* n, int type) {
    set_cost_type(n, type);
}

inline void SetOptimizerGradientDescent(neural_network* n) {
    set_optimizer(n, create_gradient_descent_optimizer());
}

inline void SetOptimizerMomentum(neural_network* n, double momentum) {
    set_optimizer(n, create_momentum_gradient_descent_optimizer(momentum));
}

inline void SetOptimizerNesterov(neural_network* n, double decay) {
    set_optimizer(n, create_nesterov_optimizer(decay));
}

inline void SetOptimizerAdam(neural_network* n, double delta1, double delta2) {
    set_optimizer(n, create_adam_optimizer(delta1, delta2));
}

inline void SetDataSelectorFullBatch(neural_network* n) {
    set_data_selector(n, create_full_batch_selector());
}

inline void SetDataSelectorMiniBatch(neural_network* n, int batchSize) {
    set_data_selector(n, create_mini_batch_selector(batchSize));
}

inline void Predict(neural_network* ptr, double inputs[], int inCount, double outputs[], int outCount) {
    input_data data;
    data.count = inCount;
    data.values = inputs;

    input_data predicted;
    predicted.count = outCount;
    predicted.values = outputs;

    predict(ptr, &data, &predicted);
}

inline neural_network* FromFile(char file[]) {
    return restore(file);
}

inline void Save(neural_network* ptr, char file[]) {
    save(ptr, file);
}

inline void Initialize(neural_network* ptr) {
    init_random();
    initialize(ptr);
}

inline void Learn(neural_network* ptr, learning_state* state, double* inputs, int inputCutoff,
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

    test_data* test = alloc_flattened_test_data(inputs, inputCutoff, expected, expectedCutoff, count);
    iterative_learn(ptr, test, state, iterations);
    free_test_data(test);
}

inline double Cost(neural_network* ptr, double* inputs, int inputCount, double* expected, int expectedCount) {
    input_data i, e;
    i.count = inputCount;
    i.values = inputs;
    e.count = expectedCount;
    e.values = expected;

    printf("%i", inputCount);
    fflush(stdout);
    return cost(ptr, &i, &e);
}

inline double MultiCost(neural_network* ptr, double* inputs, int inputCutoff, double* expected, int expectedCutoff, int count) {
    test_data* test = alloc_flattened_test_data(inputs, inputCutoff, expected, expectedCutoff, count);
    const double result = avg_cost(ptr, test);
    free_test_data(test);
    return result;
}

inline learning_state* CreateState(neural_network* ptr) {
    return alloc_state(ptr);
}

inline void DisposeState(neural_network* ptr, learning_state* state) {
    free_state(ptr, state);
}