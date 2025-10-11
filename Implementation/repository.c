#include "neural_network.h"
#include "repository.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

inline neural_network* restore(const char* file){
    FILE* fptr = fopen(file, "rb");

    int size[1];
    fread(size, sizeof(int), 1, fptr);

    int* dimensions = malloc(size[0] * sizeof(int));
    fread(dimensions, sizeof(int), size[0], fptr);

    neural_network* result = alloc_network(size[0], dimensions);
    free(dimensions);

    for(int i = 0; i < result->count; i++){
        const int wCount = result->layers[i].in_count * result->layers[i].out_count;
        fread(result->layers[i].weights, sizeof(double), wCount, fptr);
        fread(result->layers[i].biases, sizeof(double), result->layers[i].out_count, fptr);
    }

    return result;
}

inline void save(const neural_network* network, const char* file){
    FILE* fptr = fopen(file, "wb");

    int n = network->count + 1;
    int count[] = { n };
    fwrite(count, sizeof(int), 1, fptr);

    int* size = malloc(n * sizeof(int));
    size[0] = network->layers[0].in_count;
    for(int i = 0; i < network->count; i++){
        size[i + 1] = network->layers[i].out_count;
    }

    fwrite(size, sizeof(int), n, fptr);
    free(size);

    for(int i = 0; i < network->count; i++){
        fwrite(network->layers[i].weights, sizeof(double), network->layers[i].in_count * network->layers[i].out_count, fptr);
        fwrite(network->layers[i].biases, sizeof(double), network->layers[i].out_count, fptr);
    }

    fclose(fptr);
}

inline void flog(char format[], ...) {
    va_list args;
    va_start(args, format);
    const int size = vsnprintf(NULL, 0, format, args);
    va_end(args);

    if (size < 0) return;

    char* buffer = malloc((size + 1) * sizeof(char));
    va_start(args, format);
    vsnprintf(buffer, size + 1, format, args);
    va_end(args);

    FILE* fptr = fopen("log.txt", "a");
    fwrite(buffer, sizeof(char), size, fptr);
    fputc('\n', fptr);
    fclose(fptr);
    free(buffer);
}
