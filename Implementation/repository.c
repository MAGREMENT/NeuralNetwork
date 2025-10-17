#include "neural_network.h"
#include "repository.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

inline neural_network* restore(const char* file){
    FILE* fptr = fopen(file, "rb");
    if (fptr == NULL) return NULL;

    int size[1];
    if (fread(size, sizeof(int), 1, fptr) != 1) return NULL;

    int* dimensions = malloc(size[0] * sizeof(int));
    if (fread(dimensions, sizeof(int), size[0], fptr) != size[0]) return NULL;

    neural_network* result = alloc_network(size[0], dimensions);
    free(dimensions);

    for(int i = 0; i < result->count; i++){
        const int wCount = result->layers[i].in_count * result->layers[i].out_count;
        if (fread(result->layers[i].weights, sizeof(double), wCount, fptr)
            != wCount) return NULL;
        if (fread(result->layers[i].biases, sizeof(double), result->layers[i].out_count, fptr)
            != result->layers[i].out_count) return NULL;
    }

    return result;
}

inline int save(const neural_network* network, const char* file){
    FILE* fptr = fopen(file, "wb");
    if (fptr == NULL) return -1;

    int n = network->count + 1;
    int count[] = { n };
    if (fwrite(count, sizeof(int), 1, fptr) != 1) return -2;

    int* size = malloc(n * sizeof(int));
    size[0] = network->layers[0].in_count;
    for(int i = 0; i < network->count; i++){
        size[i + 1] = network->layers[i].out_count;
    }

    const size_t w = fwrite(size, sizeof(int), n, fptr);
    free(size);
    if (w != n) return -2;

    for(int i = 0; i < network->count; i++){
        const int wCount = network->layers[i].in_count * network->layers[i].out_count;
        if (fwrite(network->layers[i].weights, sizeof(double), wCount, fptr)
            != wCount) return -2;
        if (fwrite(network->layers[i].biases, sizeof(double), network->layers[i].out_count, fptr)
            != network->layers[i].out_count) return -2;
    }

    fclose(fptr);
    return 0;
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
