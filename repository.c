#include "neural_network.h"
#include "repository.h"
#include <stdio.h>

inline neural_network* initialize(const char* file, params* toFill){
    FILE* fptr = fopen(file, "rb");

    int size[1];
    fread(size, sizeof(int), 1, fptr);

    int dimensions[size[0]];
    fread(dimensions, sizeof(int), size[0], fptr);

    neural_network* result = alloc_network(size[0], dimensions);
    for(int i = 0; i < result->count; i++){
        const int wCount = result->layers[i].in_count * result->layers[i].out_count;
        fread(result->layers[i].weights, sizeof(double), wCount, fptr);
        fread(result->layers[i].biases, sizeof(double), result->layers[i].out_count, fptr);
    }

    params buff;
    params* p = toFill == NULL ? &buff : toFill;

    double b1[1];
    fread(b1, sizeof(double), 1, fptr);
    p->learningRate = b1[0];
    int b2[2];
    fread(b2, sizeof(int), 2, fptr);
    p->activationType = b2[0];
    p->costType = b2[1];

    apply_params(result, *p);
    fclose(fptr);
    return result;
}

inline void save(const neural_network* network, const params* params, const char* file){
    FILE* fptr = fopen(file, "wb");

    int n = network->count + 1;
    int count[] = { n };
    fwrite(count, sizeof(int), 1, fptr);

    int size[n];
    size[0] = network->layers[0].in_count;
    for(int i = 0; i < network->count; i++){
        size[i + 1] = network->layers[i].out_count;
    }

    fwrite(size, sizeof(int), n, fptr);
    for(int i = 0; i < network->count; i++){
        fwrite(network->layers[i].weights, sizeof(double), network->layers[i].in_count * network->layers[i].out_count, fptr);
        fwrite(network->layers[i].biases, sizeof(double), network->layers[i].out_count, fptr);
    }

    if(params != NULL){
        double ln[] = {params->learningRate};
        int types[] = {params->activationType, params->costType};

        fwrite(ln, sizeof(double), 1, fptr);
        fwrite(types, sizeof(int), 2, fptr);
    }

    fclose(fptr);
}
