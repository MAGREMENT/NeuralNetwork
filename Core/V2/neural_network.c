//
// Created by zacha on 20-10-25.
//

#include "neural_network.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Util/double_util.h"
#include "Util/rand_util.h"

neural_network* alloc_neural_network(const int layerCount) {
    neural_network* result = malloc(sizeof(neural_network));
    result->layerCount = layerCount;
    result->layers = malloc(layerCount * sizeof(layer*));
    result->optimizer = NULL;

    return result;
}

void free_neural_network(neural_network* network, const int freeConstructed) {
    if (freeConstructed) {
        for (int l = 0; l < network->layerCount; l++) {
            layer* layer = network->layers[l];
            layer->vtable->free(layer);
        }

        free(network->optimizer);
    }

    free(network->layers);
    free(network);
}

inline int get_in_count(const neural_network* network) {
    return network->layers[0]->in_count;
}

inline int get_out_count(const neural_network* network) {
    return network->layers[network->layerCount - 1]->out_count;
}

static void on_predict_start(const neural_network* network) {
    for (int i = 0; i < network->layerCount; i++) {
        const layer* l = network->layers[i];
        if (l->vtable->on_predict_start != NULL) l->vtable->on_predict_start(l);
    }
}

void predict(const neural_network* network, const double* inputs, double* outputs) {
    on_predict_start(network);
    if (network->layerCount == 1) {
        network->layers[0]->vtable->forward(network->layers[0], inputs, outputs);
        return;
    }

    double* v = NULL;

    for (int i = 0; i < network->layerCount; i++) {
        const int isLast = i == network->layerCount - 1;
        const layer* l = network->layers[i];

        if (isLast) {
            l->vtable->forward(l, v, outputs);
            free(v);
            return;
        }

        if (i == 0) {
            v = malloc(sizeof(double) * l->out_count);
            l->vtable->forward(l, inputs, v);
        } else if (l->in_count != l->out_count) {
            double* temp = malloc(sizeof(double) * l->out_count);
            l->vtable->forward(l, v, temp);
            free(v);
            v = temp;
        } else {
            l->vtable->forward(l, v, v);
        }
    }
}

static double** alloc_layer_output_buffers(const neural_network* network) {
    double** result = malloc(sizeof(double*) * network->layerCount);

    for (int i = 0; i < network->layerCount; i++) {
        result[i] = malloc(sizeof(double) * network->layers[i]->out_count);
    }

    return result;
}

double** alloc_gradient_buffers(const neural_network* network, const int initToZero) {
    double** gradients = malloc(sizeof(double*) * network->layerCount);

    for (int i = 0; i < network->layerCount; i++) {
        const layer* l = network->layers[i];
        gradients[i] = l->parameters_count <= 0 ? NULL : malloc(sizeof(double) * l->parameters_count);
        if (initToZero) memset(gradients[i], 0, l->parameters_count * sizeof(double));
    }

    return gradients;
}

static void set_gradient_buffers_to_zero(const neural_network* network, double** gradients) {
    for (int i = 0; i < network->layerCount; i++) {
        memset(gradients[i], 0, network->layers[i]->parameters_count * sizeof(double));
    }
}

void free_buffers(const neural_network* network, double** buffers) {
    for (int i = 0; i < network->layerCount; i++) {
        free(buffers[i]);
    }

    free(buffers);
}

static void average_gradients(const neural_network* network, double** buffers, const int count) {
    for (int i = 0; i < network->layerCount; i++) {
        for (int o = 0; o < network->layers[i]->parameters_count; o++) {
            buffers[i][o] /= count;
        }
    }
}

static void get_gradients(const neural_network* network, const test_data data, const iteration_range range, learning_state* state) {
    const int lastIndex = network->layerCount - 1;
    const int in_count = get_in_count(network);
    const int out_count = get_out_count(network);

    double** intermediateValues = state->iv_buffers;

    for (int r = range.from; r < range.to; r++) {
        const double* inputs = data.inputs + r * in_count;
        const double* expected = data.expected + r * out_count;

        //forward pass
        for (int i = 0; i < network->layerCount; i++) {
            const layer* l = network->layers[i];
            const double* in = i == 0 ? inputs : intermediateValues[i - 1];
            l->vtable->forward(l, in, intermediateValues[i]);
        }

        double* currentDeltas = malloc(sizeof(double) * out_count);
        network->cost_vtable->get_cost_deltas(intermediateValues[lastIndex], expected, currentDeltas, out_count);

        //backward pass
        for (int i = lastIndex; i >= 0; i--) {
            const layer* l = network->layers[i];

            if (l->parameters_count > 0) {
                const double* in = i == 0 ? inputs : intermediateValues[i - 1];
                l->vtable->deltas_to_gradients(l, in, currentDeltas, state->gradient_buffers[i]);
            }

            if (i == 0) break;

            if (l->in_count == l->out_count) l->vtable->backward(l, intermediateValues[i - 1], currentDeltas, currentDeltas);
            else {
                double* buffer = malloc(sizeof(double) * l->in_count);
                l->vtable->backward(l, intermediateValues[i - 1], currentDeltas, buffer);
                free(currentDeltas);
                currentDeltas = buffer;
            }
        }
    }
}

static void apply_gradients(const neural_network* network, const iteration_range range, const learning_state* state, const double learningRate) {
    average_gradients(network, state->gradient_buffers, range.to - range.from);
    optimizer_args opt_args = {learningRate, 0, range.iteration, state->optimizerState};

    for (int i = 0; i < network->layerCount; i++) {
        const double* g = state->gradient_buffers[i];
        if (g == NULL) continue;

        const layer* l = network->layers[i];
        if (l->parameters_count > 0) {
            opt_args.layerIndex = i;
            network->optimizer->vtable->apply_gradients(network->optimizer, l->parameters, g, l->parameters_count, opt_args);
        }
    }
}

static void on_learn_start(const neural_network* network) {
    for (int i = 0; i < network->layerCount; i++) {
        const layer* l = network->layers[i];
        if (l->vtable->on_learn_start != NULL) l->vtable->on_learn_start(l);
    }
}

void learn(const neural_network* network, const test_data data, const iteration_range range, learning_state* state, const double learningRate) {
    on_learn_start(network);
    set_gradient_buffers_to_zero(network, state->gradient_buffers);

    get_gradients(network, data, range, state);
    apply_gradients(network, range, state, learningRate);
}

void learn_stateless(const neural_network* network, const test_data data, const iteration_range range, double learningRate) {
    learning_state* state = alloc_state(network);
    learn(network, data, range, state, learningRate);
    free_state(network, state);
}

void shuffle_test_data(test_data test, const neural_network* network, const int times) {
    const int in_count = get_in_count(network);
    const int out_count = get_out_count(network);

    for (int c = 0; c < times; c++) {
        for (int i = 0; i < test.count; i++) {
            const int other = rand_i(test.count);

            for (int n = 0; n < in_count; n++) {
                const double buffer = test.inputs[i * in_count + n];
                test.inputs[i * in_count + n] = test.inputs[other * in_count + n];
                test.inputs[other * in_count + n] = buffer;
            }

            for (int n = 0; n < out_count; n++) {
                const double buffer = test.expected[i * out_count + n];
                test.expected[i * out_count + n] = test.expected[other * out_count + n];
                test.expected[other * out_count + n] = buffer;
            }
        }
    }
}

void iterative_learn(const neural_network* network, const test_data data, learning_state* state, const int iterations) {
    range_iterator* iterator = network->data_selector->vtable.cnstr_iterator(network->data_selector, data.count, iterations);
    int lastIteration = state->iteration;
    while (iterator->next(iterator)) {
        const int currentIteration = iterator->current.iteration + state->iteration;
        const double learningRate = network->scheduler->vtable->schedule(network->scheduler, network->learningRate, currentIteration);

        if (network->shuffleDataOnIteration && currentIteration != lastIteration) {
            shuffle_test_data(data, network, 1);
            lastIteration = currentIteration;
        }

        learn(network, data, iterator->current, state, learningRate);
    }

    iterator->free(iterator);
    state->iteration += iterations;
}

void iterative_learn_stateless(const neural_network* network, const test_data data, const int iterations) {
    learning_state* state = alloc_state(network);
    iterative_learn(network, data, state, iterations);
    free_state(network, state);
}

learning_state* alloc_state(const neural_network* network) {
    learning_state* state = malloc(sizeof(learning_state));
    state->iteration = 0;
    state->optimizerState = network->optimizer->vtable->cnstr_state(network->optimizer, network);

    state->gradient_buffers = alloc_gradient_buffers(network, 0);
    state->iv_buffers = alloc_layer_output_buffers(network);

    return state;
}

void free_state(const neural_network* network, learning_state* state) {
    network->optimizer->vtable->free_state(state->optimizerState, network);
    free_buffers(network, state->gradient_buffers);
    free_buffers(network, state->iv_buffers);
    free(state);
}

void initialize(const neural_network* network) {
    for (int i = 0; i < network->layerCount; i++) {
        const layer* l = network->layers[i];
        l->initialize(l);
    }
}

double get_cost(const neural_network* network, const double* inputs, const double* expected) {
    const int c = get_out_count(network);
    double* predicted = malloc(sizeof(double) * c);
    predict(network, inputs, predicted);

    const double cost = network->cost_vtable->get_cost(predicted, expected, c);

    free(predicted);
    return cost;
}

double get_avg_cost(const neural_network* network, test_data data) {
    const int in_count = get_in_count(network);
    const int out_count = get_out_count(network);
    double cost = 0;

    for (int i = 0; i < data.count; i++) {
        cost += get_cost(network, data.inputs + i * in_count, data.expected + i * out_count);
    }

    return cost / data.count;
}

double get_binary_accuracy(const neural_network* network, const test_data test) {
    const int in_count = get_in_count(network);
    const int out_count = get_out_count(network);
    double acc = 0;
    double* predicted = malloc(sizeof(double) * out_count);

    for (int i = 0; i < test.count; i++) {
        predict(network, test.inputs + i * in_count, predicted);

        const double* expected = test.expected + i * out_count;
        int ok = 1;
        for (int o = 0; o < out_count; o++) {
            if ((expected[o] >= 0.5 && predicted[o] < 0.5) || (expected[o] < 0.5 && predicted[o] >= 0.5)) {
                ok = 0;
                break;
            }
        }

        if (ok) acc++;
    }

    return acc / test.count * 100;
}

double get_classification_accuracy(const neural_network* network, const test_data test) {
    const int in_count = get_in_count(network);
    const int out_count = get_out_count(network);
    double acc = 0;
    double* predicted = malloc(sizeof(double) * out_count);

    for (int i = 0; i < test.count; i++) {
        predict(network, test.inputs + i * in_count, predicted);

        const double* expected = test.expected + i * out_count;
        if (d_max_ind(predicted, out_count) == d_max_ind(expected, out_count)) acc++;
    }

    return acc / test.count * 100;
}

inline void separate_test_data(const test_data original, const int inCutoff, const int outCutoff, test_data* training, test_data* testing, const double split) {
    const int trainingCount = (int)round(split * original.count);
    const int testCount =  original.count - trainingCount;

    training->inputs = original.inputs;
    training->expected = original.expected;
    training->count = trainingCount;

    testing->inputs = original.inputs + trainingCount * inCutoff;
    testing->expected = original.expected + trainingCount * outCutoff;
    testing->count = testCount;
}

int save_parameters(const neural_network* network, const char* file) {
    FILE* fptr = fopen(file, "wb");
    int r = 0;

    if (fptr == NULL) goto esc;

    for (int l = 0; l < network->layerCount; l++) {
        const layer* layer = network->layers[l];
        const int size[] = {layer->parameters_count};

        size_t result = fwrite(size, sizeof(int), 1, fptr);
        if (result != 1)  goto esc;

        if (layer->parameters_count == 0) continue;

        result = fwrite(layer->parameters, sizeof(double), layer->parameters_count, fptr);
        if (result != layer->parameters_count) goto esc;
    }

    r = 1;

    esc :

    fclose(fptr);
    return r;
}

int restore_parameters(const neural_network* network, const char* file) {
    FILE* fptr = fopen(file, "rb");
    int r = 0;

    if (fptr == NULL) goto esc;

    for (int l = 0; l < network->layerCount; l++) {
        const layer* layer = network->layers[l];
        int size[] = {layer->parameters_count};

        size_t result = fread(size, sizeof(int), 1, fptr);
        if (result != 1 || size[0] != layer->parameters_count) goto esc;

        if (layer->parameters_count == 0) continue;

        result = fread(layer->parameters, sizeof(double), layer->parameters_count, fptr);
        if (result != layer->parameters_count) goto esc;
    }

    r = 1;

    esc :

    fclose(fptr);
    return r;
}
