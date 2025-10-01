#include "export.h"

#include "repository.h"
#include "utils.h"

inline neural_network* Create(int count, int numbers[]) {
    return alloc_network(count, numbers);
}

inline void Dispose(neural_network* ptr) {
    free_network(ptr);
}

inline void ApplyParams(neural_network* ptr, params p) {
    apply_params(ptr, p);
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

inline void Predict(neural_network* ptr, double inputs[], int inCount, double outputs[], int outCount) {
    input_data data;
    data.count = inCount;
    data.values = inputs;

    input_data predicted;
    predicted.count = outCount;
    predicted.values = outputs;

    predict(ptr, &data, &predicted);
}

inline neural_network* FromFile(char file[], params* toFill) {
    return restore(file, toFill);
}

inline void Save(neural_network* ptr, params p, char file[]) {
    save(ptr, &p, file);
}

inline void Initialize(neural_network* ptr) {
    init_random();
    initialize(ptr);
}

inline void Learn(neural_network* ptr, learning_state* state, double* inputs, int inputCutoff,
        double* expected, int expectedCutoff, int count, int iterations) {
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

inline void DisposeState(learning_state* ptr, int layerCount) {
    free_state(ptr, layerCount);
}