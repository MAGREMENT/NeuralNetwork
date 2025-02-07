#include "export.h"
#include "repository.h"
#include "utils.h"

inline neural_network* Initialize(int count, int numbers[]) {
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

double GetBias(neural_network* ptr, int layer, int output) {
    return ptr->layers[layer].biases[output];
}

inline void Predict(neural_network* ptr, double inputs[], int inCount, double outputs[], int outCount) {
    input_data data;
    data.count = inCount;
    data.values = inputs;

    input_data* predicted = predict(ptr, &data);
    for(int i = 0; i < outCount; i++) {
        outputs[i] = predicted->values[i];
    }

    free_input_data(predicted);
}

inline neural_network* FromFile(char file[], params* toFill) {
    return initialize(file, toFill);
}

inline void Save(neural_network* ptr, params p, char file[]) {
    save(ptr, &p, file);
}

void Randomize(neural_network* ptr, double min, double max) {
    init_random();
    randomize(ptr, min, max);
}