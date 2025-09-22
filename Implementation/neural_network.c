#include <stdlib.h>
#include "neural_network.h"

#include "functions.h"
#include "utils.h"

inline neural_network* alloc_network(const int count, const int numbers[]){
    neural_network* result = malloc(sizeof(neural_network));
    result->count = count - 1;
    result->layers = malloc(result->count * sizeof(layer));

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

inline void apply_params(neural_network* network, params params){
    network->initialLearningRate = params.initialLearningRate;
    network->learningRateDecay = params.learningRateDecay;
    network->momentum = params.momentum;
    network->regularization = params.regularization;

    for(int i = 0; i < network->count; i++) {
        int type = i == network->count - 1 ? params.outputActivationType : params.activationType;
        switch (type) {
            case DEFAULT:network->layers[i].activation = default_activation;
            network->layers[i].activationDerivative = derivative_default_activation;
            network->layers[i].processInputs = default_process_inputs;
            network->layers[i].freeData = default_free_data;
            break;
            case SIGMOID : network->layers[i].activation = sigmoid_activation;
            network->layers[i].activationDerivative = derivative_sigmoid_activation;
            network->layers[i].processInputs = default_process_inputs;
            network->layers[i].freeData = default_free_data;
            break;
            case TANH : network->layers[i].activation = tanh_activation;
            network->layers[i].activationDerivative = derivative_tanh_activation;
            network->layers[i].processInputs = default_process_inputs;
            network->layers[i].freeData = default_free_data;
            break;
            case RELU : network->layers[i].activation = relu_activation;
            network->layers[i].activationDerivative = derivative_relu_activation;
            network->layers[i].processInputs = default_process_inputs;
            network->layers[i].freeData = default_free_data;
            break;
            case SILU : network->layers[i].activation = silu_activation;
            network->layers[i].activationDerivative = derivative_silu_activation;
            network->layers[i].processInputs = default_process_inputs;
            network->layers[i].freeData = default_free_data;
            break;
            case CUBE : network->layers[i].activation = cube_activation;
            network->layers[i].activationDerivative = derivative_cube_activation;
            network->layers[i].processInputs = default_process_inputs;
            network->layers[i].freeData = default_free_data;
            break;
            case SOFTMAX : network->layers[i].activation = softmax_activation;
            network->layers[i].activationDerivative = derivative_softmax_activation;
            network->layers[i].processInputs = softmax_process_inputs;
            network->layers[i].freeData = softmax_free_data;
            break;
            default: network->layers[i].activation = NULL;
            network->layers[i].activationDerivative = NULL;
            network->layers[i].processInputs = NULL;
            network->layers[i].freeData = NULL;
            break;
        }
    }

    switch (params.costType) {
        case MEAN_SQUARED:
            network->cost = mean_square_cost;
            network->costDerivative = derivative_mean_square_cost;
            break;
        default:
            network->cost = NULL;
            network->costDerivative = NULL;
    }
}

inline void randomize(neural_network* network, double min, double max) {
    for(int n = 0; n < network->count; n++) {
        for(int o = 0; o < network->layers[n].out_count; o++) {
            for(int i = 0; i < network->layers[n].in_count; i++) {
                network->layers[n].weights[i * network->layers[n].out_count + o] = random(min, max);
            }

            network->layers[n].biases[o] = random(min, max);
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

inline input_data* alloc_input_datas(int innerCount, int count){
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

void free_input_datas(input_data* data, int count) {
    for (int i = 0; i < count; i++) {
        free(data[i].values);
    }

    free(data);
}

inline void set_input_data(input_data data, const double values[]){
    for(int i = 0; i < data.count; i++){
        data.values[i] = values[i];
    }
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

        if (!isLast) {
            data = output;
        }

        if (i > 0) {
            free_input_data(data);
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

inline double multi_cost(neural_network* network, test_data* data) {
    double c = 0;
    for(int i = 0; i < data->count; i++) {
        c += cost(network, &data->inputs[i], &data->expected[i]);
    }

    return c;
}

inline layer_data* alloc_layer_data_array(neural_network* network, int copyValues) {
    layer_data* result = malloc(network->count * sizeof(layer_data));

    for(int n = 0; n < network->count; n++) {
        const int in = network->layers[n].in_count;
        const int out = network->layers[n].out_count;

        result[n].biases = malloc(out * sizeof(double));
        result[n].weights = malloc(in * out * sizeof(double));

        for(int o = 0; o < out; o++) {
            for(int i = 0; i < in; i++) {
                result[n].weights[i * out + o] = copyValues ? network->layers[n].weights[i * out + o] : 0;
            }

            result[n].biases[o] = copyValues ? network->layers[n].biases[0] : 0;
        }
    }

    return result;
}

inline void free_layer_data_array(layer_data* gradients, const int count) {
    for(int i = 0; i < count; i++){
        free(gradients[i].biases);
        free(gradients[i].weights);
    }

    free(gradients);
}

inline void apply_gradients(layer to, layer_data gradients, double learningRate){
    for(int i = 0; i < to.in_count; i++){
        for(int j = 0; j < to.out_count; j++){
            const int index = i * to.out_count + j;
            to.weights[index] -= gradients.weights[index] * learningRate;
        }
    }

    for(int i = 0; i < to.out_count; i++){
        to.biases[i] -= gradients.biases[i] * learningRate;
    }
}

inline void apply_gradients_with_velocities(layer to, layer_data gradients, layer_data velocities, double learningRate,
                                            const double momentum, const double regularization){

    const double weightDecay = 1 - regularization * learningRate;

    for(int i = 0; i < to.in_count; i++){
        for(int j = 0; j < to.out_count; j++){
            const int index = i * to.out_count + j;
            const double velocity = velocities.weights[index] * momentum - gradients.weights[index] * learningRate;

            velocities.weights[index] = velocity;
            to.weights[index] = to.weights[index] * weightDecay + velocity;
        }
    }

    for(int i = 0; i < to.out_count; i++){
        const double velocity = velocities.biases[i] * momentum - gradients.biases[i] * learningRate;

        velocities.biases[i] = velocity;
        to.biases[i] += velocity;
    }
}

inline void update_gradients(const neural_network* network, const layer_data* gradients, input_data input,
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

inline void learn(neural_network* network, test_data* data, batch batch, double learningRate, layer_data* velocities){
    layer_data* gradients = alloc_layer_data_array(network, 0);

    for(int i = batch.then; i < batch.to; i++){
        update_gradients(network, gradients, data->inputs[i], data->expected[i]);
    }

    for(int i = 0; i < batch.then; i++) {
        update_gradients(network, gradients, data->inputs[i], data->expected[i]);
    }

    if (velocities == NULL) {
        for(int i = 0; i < network->count; i++){
            apply_gradients(network->layers[i], gradients[i], learningRate);
        }
    }
    else {
        for(int i = 0; i < network->count; i++){
            apply_gradients_with_velocities(network->layers[i], gradients[i], velocities[i], learningRate,
                network->momentum, network->regularization);
        }
    }

    free_layer_data_array(gradients, network->count);
}

inline void iterative_learn(neural_network* network, test_data* data, const int batchSize, const int iterations) {
    layer_data* velocities = network->regularization == 0 && network->momentum == 0
        ? NULL
        : alloc_layer_data_array(network, false);

    const double baseLr = network->initialLearningRate / batchSize;
    double learningRate = baseLr;
    int current = 0;

    for(int iteration = 0; iteration < iterations; iteration++) {
        const batch b = create_batch(current, batchSize, data->count);
        learn(network, data, b, learningRate, velocities);

        current = b.then == 0 ? b.to : b.then;
        learningRate = 1.0 / (1.0 + network->learningRateDecay * iteration) * baseLr;
    }

    if (velocities != NULL) free_layer_data_array(velocities, network->count);
}

inline int is_valid(input_data* output, input_data* expected) {
    return max_index(output->values, output->count) == max_index(expected->values, expected->count);
}

inline test_data* alloc_test_data(const int count, const int inputCount, const int outputCount) {
    test_data* result = malloc(sizeof(test_data));
    result->count = count;
    result->inputs = alloc_input_datas(inputCount, count);
    result->expected = alloc_input_datas(outputCount, count);

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
    free_input_datas(data->inputs, data->count);
    free_input_datas(data->expected, data->count);
    free(data);
}

inline test_result test_network(neural_network* network, test_data *test) {
    double valid = 0;
    for(int i = 0; i < test->count; i++) {
        input_data* output = alloc_predict(network, &test->inputs[i]);
        if(is_valid(output, &test->expected[i])) valid++;
        free_input_data(output);
    }

    test_result result;
    result.cost = multi_cost(network, test);
    result.accuracy = valid / test->count * 100;

    return result;
}

inline batch create_batch(const int current, const int size, const int max) {
    batch result;
    result.from = current;

    if(size == max) {
        result.to = max;
        result.then = current;
        return result;
    }

    const int to = current + size;
    if(to >= max) {
        result.to = max;
        result.then = to % max;
    }
    else {
        result.to = to;
        result.then = 0;
    }

    return result;
}

inline batch full_batch(const int max) {
    batch result;
    result.from = 0;
    result.to = max;
    result.then = 0;

    return result;
}
