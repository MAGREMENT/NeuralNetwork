#include <stdlib.h>

#include "neural_network.h"

#include <math.h>
#include <stdio.h>

#include "functions.h"
#include "multi-threading.h"
#include "store.h"
#include "utils.h"

inline neural_network* alloc_network(const int count, const int numbers[]){
    neural_network* result = malloc(sizeof(neural_network));
    result->count = count - 1;
    result->layers = malloc(result->count * sizeof(layer));

    result->optimizer = NULL;
    result->data_selector = NULL;
    result->scheduler = NULL;

    for(int i = 1; i < count; i++){
        const int in = numbers[i - 1];
        const int out = numbers[i];

        result->layers[i - 1].in_count = in;
        result->layers[i - 1].out_count = out;
        result->layers[i - 1].weights = malloc(in * out * sizeof(double));
        result->layers[i - 1].biases = malloc(out * sizeof(double));
    }

    return result;
}

static void check_output_layer_delta_optimization(neural_network* network) {
    network->outputLayerDeltaOptimization =
        network->layers[network->count - 1].activationDerivative == derivative_softmax_activation
        && network->costDerivative == derivative_binary_cross_entropy_cost;
}

inline void set_activation_type(neural_network* network, int type, int outputType) {
    for(int i = 0; i < network->count; i++) {
        const int t = i == network->count - 1 ? outputType : type;
        if (t < 0 || t > activation_store_max) {
            network->layers[i].activation = NULL;
            network->layers[i].activationDerivative = NULL;
            network->layers[i].processInputs = NULL;
            network->layers[i].freeData = NULL;
            network->layers[i].initialization = NULL;
        }
        else {
            const activation_data data = activation_store[t];
            network->layers[i].activation = data.activation;
            network->layers[i].activationDerivative = data.activationDerivative;
            network->layers[i].processInputs = data.processInputs;
            network->layers[i].freeData = data.freeData;
            network->layers[i].initialization = data.initialization;
        }
    }

    check_output_layer_delta_optimization(network);
}

inline void set_cost_type(neural_network* network, int type) {
    if (type < 0 || type > cost_store_max) {
        network->cost = NULL;
        network->costDerivative = NULL;
    }
    else {
        const cost_data data = cost_store[type];
        network->cost = data.cost;
        network->costDerivative = data.costDerivative;
    }

    check_output_layer_delta_optimization(network);
}

void get_activation_type(neural_network* network, int* type, int* outputType) {
    *type = find_activation(network->layers[0].activation);
    *outputType = find_activation(network->layers[network->count - 1].activation);
}

inline int get_cost_type(neural_network* network) {
    return find_cost(network->cost);
}

void set_optimizer(neural_network* n, optimizer* opt) {
    if (n->optimizer != NULL) n->optimizer->free(n->optimizer);
    n->optimizer = opt;
}

void set_data_selector(neural_network* n, data_selector* ds) {
    if (n->data_selector != NULL) n->data_selector->free(n->data_selector);
    n->data_selector = ds;
}

void set_scheduler(neural_network* n, learning_rate_scheduler* lrs) {
    if (n->scheduler != NULL) n->scheduler->free(n->scheduler);
    n->scheduler = lrs;
}

inline void initialize(neural_network* network) {
    for (int l = 0; l < network->count; l++) {
        network->layers[l].initialization(&network->layers[l]);

        for (int o = 0; o < network->layers[l].out_count; o++) {
            network->layers[l].biases[o] = 0.0;
        }
    }
}

void set_all_weights_and_biases(neural_network* network, double weights, double biases) {
    for(int n = 0; n < network->count; n++) {
        for(int o = 0; o < network->layers[n].out_count; o++) {
            for(int i = 0; i < network->layers[n].in_count; i++) {
                network->layers[n].weights[i * network->layers[n].out_count + o] = weights;
            }

            network->layers[n].biases[o] = biases;
        }
    }
}

inline void free_layers(layer* layers, int count) {
    for(int i = 0; i < count; i++){
        free(layers[i].biases);
        free(layers[i].weights);
    }

    free(layers);
}

inline void free_network(neural_network* network){
    if (network->optimizer != NULL) network->optimizer->free(network->optimizer);
    if (network->scheduler != NULL) network->scheduler->free(network->scheduler);
    if (network->data_selector != NULL) network->data_selector->free(network->data_selector);

    free_layers(network->layers, network->count);
    free(network);
}

inline void set_layer(layer layer, const double* weights, const double* biases){
    for(int i = 0; i < layer.in_count; i++){
        for(int j = 0; j < layer.out_count; j++){
            layer.weights[i * layer.out_count + j] = weights[i * layer.out_count + j];
        }
    }

    for(int i = 0; i < layer.out_count; i++){
        layer.biases[i] = biases[i];
    }
}

inline input_data* alloc_input_data(int count){
    input_data* result = malloc(sizeof(input_data));

    result->count = count;
    result->values = malloc(count * sizeof(double));
    return result;
}

inline input_data* alloc_input_data_array(int innerCount, int count){
    input_data* result = malloc(sizeof(input_data) * count);

    for (int i = 0; i < count; i++) {
        result[i].count = innerCount;
        result[i].values = malloc(innerCount * sizeof(double));
    }
    return result;
}

inline void free_input_data(input_data* data){
    free(data->values);
    free(data);
}

void free_input_data_array(input_data* data, int count) {
    for (int i = 0; i < count; i++) {
        free(data[i].values);
    }

    free(data);
}

inline backpropagation_data* alloc_back_data(const neural_network* network) {
    backpropagation_data* result = malloc(network->count * sizeof(backpropagation_data));

    for(int i = 0; i < network->count; i++){
        const int n = network->layers[i].out_count;
        result[i].count = n;
        result[i].weightedInputs = malloc(n * sizeof(double));
        result[i].afterActivations = malloc(n * sizeof(double));
        result[i].nodeValues = malloc(n * sizeof(double));
    }

    return result;
}

inline void free_back_data(backpropagation_data* data, const int count){
    for(int i = 0; i < count; i++){
        free(data[i].weightedInputs);
        free(data[i].afterActivations);
        free(data[i].nodeValues);
    }

    free(data);
}

inline void forward(layer layer, input_data input, input_data* result) {
    double* weightedInputs = malloc(layer.out_count * sizeof(double));

    for(int o = 0; o < layer.out_count; o++){
        double n = layer.biases[o];

        for(int i = 0; i < layer.in_count; i++){
            const int ind = i * layer.out_count + o;
            n += input.values[i] * layer.weights[ind];
        }

        weightedInputs[o] = n;
    }

    void* data = layer.processInputs(weightedInputs, layer.out_count);
    for(int o = 0; o < layer.out_count; o++) {
        result->values[o] = layer.activation(weightedInputs[o], data);
    }
    layer.freeData(data);

    free(weightedInputs);
}

input_data* alloc_predict(neural_network* network, input_data* data) {
    input_data* result = alloc_input_data(network->layers[network->count - 1].out_count);
    predict(network, data, result);
    return result;
}

inline void predict(neural_network* network, input_data* data, input_data* result) {
    for(int i = 0; i < network->count; i++){
        const int isLast = i == network->count - 1;

        input_data* output = isLast ? result : alloc_input_data(network->layers[i].out_count);
        forward(network->layers[i], *data, output);

        if (i > 0) {
            free_input_data(data);
        }

        if (!isLast) {
            data = output;
        }
    }
}

inline void continue_advance(const layer layer, const backpropagation_data* data, const int inputIndex){
    for(int o = 0; o < layer.out_count; o++){
        double n = layer.biases[o];

        for(int i = 0; i < layer.in_count; i++){
            const int ind = i * layer.out_count + o;
            n += data[inputIndex].afterActivations[i] * layer.weights[ind];
        }

        data[inputIndex + 1].weightedInputs[o] = n;
    }

    void* d = layer.processInputs(data[inputIndex + 1].weightedInputs, layer.out_count);
    for(int o = 0; o < layer.out_count; o++) {
        data[inputIndex + 1].afterActivations[o] = layer.activation(data[inputIndex + 1].weightedInputs[o], d);
    }
    layer.freeData(d);
}

inline void first_advance(const layer layer, const backpropagation_data* data, const input_data* input){
    for(int o = 0; o < layer.out_count; o++){
        double n = layer.biases[o];

        for(int i = 0; i < layer.in_count; i++){
            const int ind = i * layer.out_count + o;
            n += input->values[i] * layer.weights[ind];
        }

        data[0].weightedInputs[o] = n;
    }

    void* d = layer.processInputs(data[0].weightedInputs, layer.out_count);
    for(int o = 0; o < layer.out_count; o++) {
        data[0].afterActivations[o] = layer.activation(data[0].weightedInputs[o], d);
    }
    layer.freeData(d);
}

inline void traverse(const neural_network* network, input_data* data, backpropagation_data* result){
    first_advance(network->layers[0], result, data);
    for(int i = 0; i < network->count - 1; i++) {
        continue_advance(network->layers[i + 1], result, i);
    }
}

inline backpropagation_data* alloc_traverse(const neural_network* network, input_data* data) {
    backpropagation_data* result = alloc_back_data(network);
    traverse(network, data, result);
    return result;
}

inline double cost(neural_network* network, input_data* data, input_data* expected) {
    input_data* result = alloc_predict(network, data);
    double cost = 0;
    for(int i = 0; i < expected->count; i++) {
        cost += network->cost(result->values[i], expected->values[i]);
    }

    free_input_data(result);
    return cost;
}

inline double avg_cost(neural_network* network, test_data* data) {
    if (data->count == 0) return 0;

    double c = 0;
    for(int i = 0; i < data->count; i++) {
        c += cost(network, &data->inputs[i], &data->expected[i]);
    }

    return c / data->count;
}

static void setup_nv(const backpropagation_data* data,
        const neural_network* network,
        const input_data expected) {

    const int lastIndex = network->count - 1;

    for(int l = lastIndex; l >= 0; l--) {
        void* d = network->layers[l].processInputs(data[l].weightedInputs, network->layers[l].out_count);

        if(l == lastIndex) {
            if (network->outputLayerDeltaOptimization) {
                for(int i = 0; i < expected.count; i++){
                    data[l].nodeValues[i] = data[l].afterActivations[i] - expected.values[i];
                }
            } else {
                for(int i = 0; i < expected.count; i++){
                    const double costDerivative = network->costDerivative(data[l].afterActivations[i], expected.values[i]);
                    const double activationDerivative = network->layers[l].activationDerivative(data[l].weightedInputs[i], d);
                    data[l].nodeValues[i] = activationDerivative * costDerivative;
                }
            }
        }
        else {
            const int out = network->layers[l + 1].out_count;
            const int in = network->layers[l].out_count;
            for(int i = 0; i < in; i++) {
                double value = 0;
                for(int o = 0; o < out; o++) {
                    const double w = network->layers[l + 1].weights[i * out + o];
                    const double nv = data[l + 1].nodeValues[o];
                    value += nv * w;
                }

                data[l].nodeValues[i] = value * network->layers[l].activationDerivative(data[l].weightedInputs[i], d);
            }
        }

        network->layers[l].freeData(d);
    }
}

static void add_gradients_from_nv(const backpropagation_data* data,
        const neural_network* network, const layer_data* gradients, input_data input) {

    const int lastIndex = network->count - 1;

    for(int l = lastIndex; l >= 0; l--) {
        const layer current = network->layers[l];
        for(int o = 0; o < current.out_count; o++) {
            const double nv = data[l].nodeValues[o];

            for(int i = 0; i < current.in_count; i++) {
                const double a = l == 0 ? input.values[i] : data[l - 1].afterActivations[i];
                const double g = nv * a;
                gradients[l].weights[i * current.out_count + o] += g;
            }

            gradients[l].biases[o] += nv;
        }
    }
}

inline void add_gradients(const neural_network* network, const layer_data* gradients, input_data input,
        const input_data expected) {

    backpropagation_data* data = alloc_traverse(network, &input);

    setup_nv(data, network, expected);
    add_gradients_from_nv(data, network, gradients, input);

    free_back_data(data, network->count);
}

void async_add_gradients(const neural_network* network, const layer_data* gradients, input_data input,
    input_data expected, void* criticalSection) {

    backpropagation_data* data = alloc_traverse(network, &input);

    setup_nv(data, network, expected);

    enter_critical_section(criticalSection);
    add_gradients_from_nv(data, network, gradients, input);
    exit_critical_section(criticalSection);

    free_back_data(data, network->count);
}

static void average_gradients(neural_network* network, layer_data* gradients, int dataCount) {
    for (int l = 0; l < network->count; l++) {
        const int oc = network->layers[l].out_count;
        const int ic = network->layers[l].in_count;
        for (int o = 0; o < oc; o++) {
            for (int i = 0; i < ic; i++) {
                gradients[l].weights[i * oc + o] /= dataCount;
            }

            gradients[l].biases[o] /= dataCount;
        }
    }
}

typedef struct async_gradient_computation {
    neural_network* network;
    layer_data* gradients;
    test_data* data;
    range range;
    void* section;
} async_gradient_computation;

static void add_gradients_parallel(void* params, parallel_thread_info threadInfo) {
    async_gradient_computation* agc = params;

    const int delta = (agc->range.to - agc->range.from) / threadInfo.total;
    const int start = delta * threadInfo.index;
    int end = start + delta;
    if (threadInfo.index == threadInfo.total - 1) {
        end += (agc->range.to - agc->range.from) % threadInfo.total;
    }

    for (int i = start; i < end; i++) {
        async_add_gradients(agc->network, agc->gradients, agc->data->inputs[i], agc->data->expected[i], agc->section);
    }
}

inline void learn(neural_network* network, test_data* data, range range, const double learningRate,
        void* optimizerState){
    //TODO optimize : pass as an argument when called from iterative_learn()
    layer_data* gradients = alloc_layer_data_array(network->layers, network->count, 0);

    if (network->threadCount <= 1) {
        for(int i = range.from; i < range.to; i++){
            add_gradients(network, gradients, data->inputs[i], data->expected[i]);
        }
    }
    else {
        void* cs = alloc_critical_section();
        async_gradient_computation params;

        params.network = network;
        params.gradients = gradients;
        params.data = data;
        params.range = range;
        params.section = cs;

        exec_parallel(add_gradients_parallel, &params, network->threadCount);
        free_critical_section(cs);
    }

    average_gradients(network, gradients, range.to - range.from);

    network->optimizer->apply_gradients(network->optimizer, optimizerState, network->layers, gradients,
            network->count, range.iteration, learningRate);

    free_layer_data_array(gradients, network->count);
}

void iterative_learn(neural_network* network, test_data* data, learning_state* state, int iterations) {
    int freeState = false;
    if (state == NULL) {
        state = alloc_state(network);
        freeState = true;
    }

    range_iterator* iterator = network->data_selector->constr_iterator(network->data_selector, data->count, iterations);
    int lastIteration = state->iteration;
    while (iterator->next(iterator)) {
        const int currentIteration = iterator->current.iteration + state->iteration;
        const double learningRate = network->scheduler->schedule(network->scheduler, network->learningRate, currentIteration);

        if (network->shuffleDataOnIteration && currentIteration != lastIteration) {
            shuffle_test_data(data, 1);
            lastIteration = currentIteration;
        }

        learn(network, data, iterator->current, learningRate, state->optimizerState);
    }

    iterator->free(iterator);
    if (freeState) free_state(network, state);
    else state->iteration += iterations;
}

inline learning_state* alloc_state(neural_network* network) {
    learning_state* state = malloc(sizeof(learning_state));
    state->iteration = 0;
    state->optimizerState = network->optimizer->create_state(network->optimizer, network->layers, network->count);

    return state;
}

inline void free_state(neural_network* network, learning_state* state) {
    network->optimizer->free_state(state->optimizerState, network->count);
    free(state);
}

inline int is_valid(input_data* output, input_data* expected) {
    return max_index(output->values, output->count) == max_index(expected->values, expected->count);
}

inline test_data* alloc_test_data(const int count, const int inputCount, const int outputCount) {
    test_data* result = malloc(sizeof(test_data));
    result->count = count;
    result->inputs = alloc_input_data_array(inputCount, count);
    result->expected = alloc_input_data_array(outputCount, count);

    return result;
}

inline test_data* alloc_transfer_flattened_data(double* inputs, int inputCutoff, double* expected, int expectedCutoff, int count) {
    test_data* test = malloc(sizeof(test_data));
    test->count = count;
    test->inputs = malloc(sizeof(input_data) * count);
    test->expected = malloc(sizeof(input_data) * count);

    for (int i = 0; i < count; i++) {
        test->inputs[i].count = inputCutoff;
        test->inputs[i].values = inputs + inputCutoff * i;

        test->expected[i].count = expectedCutoff;
        test->expected[i].values = expected + expectedCutoff * i;
    }

    return test;
}

inline void free_transferred_flattened_data(test_data* data) {
    free(data->inputs);
    free(data->expected);
    free(data);
}

inline void free_test_data(test_data* data){
    free_input_data_array(data->inputs, data->count);
    free_input_data_array(data->expected, data->count);
    free(data);
}

void shuffle_test_data(test_data *test, int count) {
    for (int c = 0; c < count; c++) {
        for (int i = 0; i < test->count; i++) {
            const int other = rand_i(test->count);

            input_data buffer = test->inputs[i];
            test->inputs[i] = test->inputs[other];
            test->inputs[other] = buffer;

            buffer = test->expected[i];
            test->expected[i] = test->expected[other];
            test->expected[other] = buffer;
        }
    }
}

inline test_result test_network(neural_network* network, test_data *test) {
    double valid = 0;
    for(int i = 0; i < test->count; i++) {
        input_data* output = alloc_predict(network, &test->inputs[i]);
        if(is_valid(output, &test->expected[i])) valid++;
        free_input_data(output);
    }

    test_result result;
    result.cost = avg_cost(network, test);
    result.accuracy = valid / test->count * 100;

    return result;
}

static int process_gradient_scale(gradient_diagnostic* diag, double value, int vanishingBound,
        int explodingBound) {
    value = fabs(value);
    const int x = floor(log(value) / log(10));
    const int result = x <= vanishingBound || x >= explodingBound ? x : 0;

    for (int i = 0; i < diag->scales->count; i++) {
        if (l_get(diag->scales, gradient_scale, i).lower <= x && l_get(diag->scales, gradient_scale, i).upper > x) {
            l_pget(diag->scales, gradient_scale, i)->count++;
            return result;
        }
    }

    gradient_scale s;
    s.count = 1;
    s.upper = x + 1;
    s.lower = x;
    l_add(diag->scales, gradient_scale, s);
    return result;
}

static int compare_grad_scale(void* s1, void* s2) {
    return ((gradient_scale*)s2)->lower - ((gradient_scale*)s1)->lower;
}

inline gradient_diagnostic* alloc_run_gradient_diagnostic(neural_network* network, test_data* data, int vanishingBound,
        int explodingBound) {
    gradient_diagnostic* diag = malloc(sizeof(gradient_diagnostic));
    diag->scales = alloc_list(sizeof(gradient_scale));
    diag->criticalBiases = alloc_list(sizeof(bias_scale));
    diag->criticalWeights = alloc_list(sizeof(weight_scale));

    layer_data* gradients = alloc_layer_data_array(network->layers, network->count, 0);

    for(int i = 0; i < data->count; i++){
        add_gradients(network, gradients, data->inputs[i], data->expected[i]);
    }

    average_gradients(network, gradients, data->count);

    for (int l = 0; l < network->count; l++) {
        const int out_count = network->layers[l].out_count;
        const int in_count = network->layers[l].in_count;

        for (int o = 0; o < out_count; o++) {
            for (int i = 0; i < in_count; i++) {
                const int index = i * out_count + o;
                const int scale = process_gradient_scale(diag, gradients[l].weights[index], vanishingBound, explodingBound);
                if (scale) {
                    weight_scale s;
                    s.layer = l;
                    s.in = i;
                    s.out = o;
                    s.scale = scale;
                    l_add(diag->criticalWeights, weight_scale, s);
                }
            }

            const int scale = process_gradient_scale(diag, gradients[l].biases[o], vanishingBound, explodingBound);
            if (scale) {
                bias_scale s;
                s.layer = l;
                s.out = o;
                s.scale = scale;
                l_add(diag->criticalBiases, bias_scale, s);
            }
        }
    }

    free_layer_data_array(gradients, network->count);

    qsort(diag->scales->data, diag->scales->count, sizeof(gradient_scale), compare_grad_scale);
    return diag;
}

inline void free_gradient_diagnostic(gradient_diagnostic* diag) {
    free_list(diag->scales);
    free_list(diag->criticalBiases);
    free_list(diag->criticalWeights);
    free(diag);
}

inline void print_diagnostic(neural_network* network, gradient_diagnostic* diag) {
    for (int i = 0; i < diag->scales->count; i++) {
        gradient_scale s = l_get(diag->scales, gradient_scale, i);
        printf("%d gradients more than %d and less than %d\n",
            s.count, s.lower, s.upper);
    }

    for (int i = 0; i < diag->criticalWeights->count; i++) {
        weight_scale w = l_get(diag->criticalWeights, weight_scale, i);
        printf("Critical gradient l%d i%d o%d with value %f and scale %d\n",
            w.layer, w.in, w.out, network->layers[w.layer].weights[w.in * network->layers[w.layer].out_count + w.out], w.scale);
    }

    for (int i = 0; i < diag->criticalBiases->count; i++) {
        bias_scale b = l_get(diag->criticalBiases, bias_scale, i);
        printf("Critical gradient l%d o%d with value %f and scale %d\n",
            b.layer, b.out, network->layers[b.layer].biases[b.out], b.scale);
    }
}
