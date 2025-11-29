//
// Created by zacha on 29-11-25.
//

#include "export.h"

builder* CreateBuilder() {
    return alloc_builder(0);
}

void LoadBuilderFile(builder* builder, char file[]) {
    yaml_reader* reader = alloc_yaml_reader();
    download_yaml(reader, file);
    from_yaml(builder, reader);
    free_yaml_reader(reader);
}

neural_network* Build(builder* builder) {
    return build(builder, def_b_params());
}

void DisposeBuilder(builder* builder) {
    free_builder(builder);
}

void Dispose(neural_network* ptr) {
    free_neural_network(ptr, 1);
}

int GetLength(neural_network* ptr) {
    return ptr->layerCount;
}

int GetInCount(neural_network* ptr) {
    return get_in_count(ptr);
}

int GetOutCount(neural_network* ptr) {
    return get_out_count(ptr);
}

void LoadParameterFile(neural_network* ptr, char file[]) {
    restore_parameters(ptr, file);
}

void Predict(neural_network* ptr, double* inputs, double* outputs) {
    predict(ptr, inputs, outputs);
}

void Learn(neural_network* ptr, learning_data* state, double* inputs, double* expected, int count, int iterations) {
    const test_data data = {inputs, expected, count};
    iterative_learn(ptr, data, state, iterations);
}

void LearnStateless(neural_network* ptr, double* inputs, double* expected, int count, int iterations) {
    const test_data data = {inputs, expected, count};
    iterative_learn_stateless(ptr, data, iterations);
}

double Cost(neural_network* ptr, double* inputs, double* expected, int count) {
    const test_data data = {inputs, expected, count};
    return get_avg_cost(ptr, data);
}

learning_data* CreateState(neural_network* ptr) {
    return alloc_learning_data(ptr);
}

void DisposeState(neural_network* ptr, learning_data* state) {
    free_learning_data(ptr, state);
}