#include <stdlib.h>

#include "neural_network.h"

#include <math.h>

#include "functions.h"
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

inline void set_activation_type(neural_network* network, int type, int outputType) {
    for(int i = 0; i < network->count; i++) {
        const int t = i == network->count - 1 ? outputType : type;
        switch (t) {
            case DEFAULT :
                network->layers[i].activation = default_activation;
                network->layers[i].activationDerivative = derivative_default_activation;
                network->layers[i].processInputs = default_process_inputs;
                network->layers[i].freeData = default_free_data;

                if (i == 0) network->initialization = random_initialization;
            break;
            case SIGMOID :
                network->layers[i].activation = sigmoid_activation;
                network->layers[i].activationDerivative = derivative_sigmoid_activation;
                network->layers[i].processInputs = default_process_inputs;
                network->layers[i].freeData = default_free_data;

                if (i == 0) network->initialization = xavier_initialization;
            break;
            case TANH :
                network->layers[i].activation = tanh_activation;
                network->layers[i].activationDerivative = derivative_tanh_activation;
                network->layers[i].processInputs = default_process_inputs;
                network->layers[i].freeData = default_free_data;

                if (i == 0) network->initialization = xavier_initialization;
            break;
            case RELU :
                network->layers[i].activation = relu_activation;
                network->layers[i].activationDerivative = derivative_relu_activation;
                network->layers[i].processInputs = default_process_inputs;
                network->layers[i].freeData = default_free_data;

                if (i == 0) network->initialization = he_initialization;
            break;
            case LEAKY_RELU :
                network->layers[i].activation = leaky_relu_activation;
                network->layers[i].activationDerivative = derivative_leaky_relu_activation;
                network->layers[i].processInputs = default_process_inputs;
                network->layers[i].freeData = default_free_data;

                if (i == 0) network->initialization = he_initialization;
            break;
            case SILU :
                network->layers[i].activation = silu_activation;
                network->layers[i].activationDerivative = derivative_silu_activation;
                network->layers[i].processInputs = default_process_inputs;
                network->layers[i].freeData = default_free_data;

                if (i == 0) network->initialization = random_initialization;
            break;
            case SOFTMAX :
                network->layers[i].activation = softmax_activation;
                network->layers[i].activationDerivative = derivative_softmax_activation;
                network->layers[i].processInputs = softmax_process_inputs;
                network->layers[i].freeData = softmax_free_data;

                if (i == 0) network->initialization = random_initialization;
            break;
            default:
                network->layers[i].activation = NULL;
                network->layers[i].activationDerivative = NULL;
                network->layers[i].processInputs = NULL;
                network->layers[i].freeData = NULL;

                if (i == 0) network->initialization = NULL;
            break;
        }
    }
}

inline void set_cost_type(neural_network* network, int type) {
    switch (type) {
        case MEAN_SQUARED:
            network->cost = mean_square_cost;
            network->costDerivative = derivative_mean_square_cost;
        break;
        case CROSS_ENTROPY:
            network->cost = cross_entropy_cost;
            network->costDerivative = derivative_cross_entropy_cost;
        break;
        default:
            network->cost = NULL;
            network->costDerivative = NULL;
        break;
    }
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
        network->initialization(&network->layers[l]);

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
    network->optimizer->free(network->optimizer);
    network->scheduler->free(network->scheduler);
    network->data_selector->free(network->data_selector);

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

inline void add_gradients(const neural_network* network, const layer_data* gradients, input_data input,
        const input_data expected) {

    backpropagation_data* data = alloc_traverse(network, &input);
    const int lastIndex = network->count - 1;

    for(int n = lastIndex; n >= 0; n--) {
        void* d = network->layers[n].processInputs(data[n].weightedInputs, network->layers[n].out_count);

        if(n == lastIndex) {
            for(int i = 0; i < expected.count; i++){
                const double costDerivative = network->costDerivative(data[n].afterActivations[i], expected.values[i]);
                const double activationDerivative = network->layers[n].activationDerivative(data[n].weightedInputs[i], d);
                data[n].nodeValues[i] = activationDerivative * costDerivative;
            }
        }
        else {
            const int out = network->layers[n + 1].out_count;
            for(int i = 0; i < network->layers[n].out_count; i++) {
                double value = 0;
                for(int o = 0; o < out; o++) {
                    const double w = network->layers[n + 1].weights[i * out + o];
                    const double nv = data[n + 1].nodeValues[o];
                    value += nv * w;
                }

                data[n].nodeValues[i] = value * network->layers[n].activationDerivative(data[n].weightedInputs[i], d);
            }
        }

        network->layers[n].freeData(d);

        const layer current = network->layers[n];
        for(int o = 0; o < current.out_count; o++) {
            const double nv = data[n].nodeValues[o];

            for(int i = 0; i < current.in_count; i++) {
                const double g = nv * (n == 0 ? input.values[i] : data[n - 1].afterActivations[i]);
                gradients[n].weights[i * current.out_count + o] += g;
            }

            gradients[n].biases[o] += nv;
        }
    }

    free_back_data(data, network->count);
}

static void normalize_gradients(neural_network* network, layer_data* gradients, int dataCount) {
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

inline void learn(neural_network* network, test_data* data, range range, const double learningRate, void* optimizerState){
    layer_data* gradients = alloc_layer_data_array(network->layers, network->count, 0);

    for(int i = range.from; i < range.to; i++){
        add_gradients(network, gradients, data->inputs[i], data->expected[i]);
    }

    normalize_gradients(network, gradients, range.to - range.from);

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

inline test_data* alloc_flattened_test_data(double* inputs, int inputCutoff, double* expected, int expectedCutoff, int count) {
    test_data* test = alloc_test_data(count, inputCutoff, expectedCutoff);

    for (int i = 0; i < count; i++) {
        int start = inputCutoff * i;
        test->inputs[i].count = inputCutoff;

        for (int j = 0; j < inputCutoff; j++) {
            test->inputs[i].values[j] = inputs[start + j];
        }

        start = expectedCutoff * i;
        test->expected[i].count = expectedCutoff;

        for (int j = 0; j < expectedCutoff; j++) {
            test->expected[i].values[j] = expected[start + j];
        }
    }

    return test;
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

static void process_gradient_stat(gradient_diagnostic* diag, double value, int vanishingBound,
        int explodingBound) {
    value = fabs(value);
    const int x = floor(log(value) / log(10));

    for (int i = 0; i < diag->count; i++) {
        if (diag->scales[i].lower <= x && diag->scales[i].upper > x) {
            diag->scales[i].count++;
            return;
        }
    }

    diag->scales = list_grow(diag->scales, sizeof(gradient_scale), diag->count, diag->count + 1);
    diag->count += 1;

    const int ind = diag->count - 1;
    diag->scales[ind].count = 1;
    diag->scales[ind].upper = x + 1;
    diag->scales[ind].lower = x;
}

static int compare_grad_scale(void* s1, void* s2) {
    return ((gradient_scale*)s2)->lower - ((gradient_scale*)s1)->lower;
}

inline gradient_diagnostic* alloc_run_gradient_diagnostic(neural_network* network, test_data* data, int vanishingBound,
        int explodingBound) {
    gradient_diagnostic* diag = malloc(sizeof(gradient_diagnostic));
    diag->count = 0;
    diag->scales = NULL;

    layer_data* gradients = alloc_layer_data_array(network->layers, network->count, 0);

    for(int i = 0; i < data->count; i++){
        add_gradients(network, gradients, data->inputs[i], data->expected[i]);
    }

    normalize_gradients(network, gradients, data->count);

    for (int l = 0; l < network->count; l++) {
        const int out_count = network->layers[l].out_count;
        const int in_count = network->layers[l].in_count;

        for (int o = 0; o < in_count; o++) {
            for (int i = 0; i < in_count; i++) {
                const int index = i * out_count + in_count;
                process_gradient_stat(diag, gradients[l].weights[index], vanishingBound, explodingBound);
            }

            process_gradient_stat(diag, gradients[l].biases[o], vanishingBound, explodingBound);
        }
    }

    free_layer_data_array(gradients, network->count);

    qsort(diag->scales, diag->count, sizeof(gradient_scale), compare_grad_scale);
    return diag;
}

inline void free_gradient_diagnostic(gradient_diagnostic* diag) {
    free(diag->scales);
    free(diag);
}
