//
// Created by zacha on 07-11-25.
//

#include "cuda_dense_layer.cuh"
#include <cuda_runtime.h>

#include "../dense_layer.h"

typedef struct cuda_dense_layer_params {
    double* gpu_in;
    double* gpu_out;
    double* gpu_parameters;
    int threads;
    int blocks;
} cuda_dense_layer_params;

__global__ void kernel_dense_forward(const double *in, const double *p, double *out, const int in_count, const int out_count) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    const double* b = p + in_count * out_count;

    if (o < out_count) {
        double v = b[o];

        for(int i = 0; i < in_count; i++){
            const int ind = i * out_count + o;
            v += in[i] * p[ind];
        }

        out[o] = v;
    }
}

static void cuda_dense_forward(const layer* l, const double* inputs, double* outputs) {
    const auto p = (cuda_dense_layer_params*)l->data;

    cudaMemcpy(p->gpu_in, inputs, l->in_count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(p->gpu_parameters, l->parameters, l->parameters_count * sizeof(double), cudaMemcpyHostToDevice);

    kernel_dense_forward<<<p->blocks, p->threads>>>(p->gpu_in, p->gpu_parameters, p->gpu_out, l->in_count, l->out_count);

    cudaDeviceSynchronize();
    cudaMemcpy(outputs, p->gpu_out, l->out_count * sizeof(double), cudaMemcpyDeviceToHost);
}

static void free_cuda_dense_layer(layer* l) {
    const auto p = (cuda_dense_layer_params*)l->data;

    cudaFree(p->gpu_in);
    cudaFree(p->gpu_out);
    cudaFree(p->gpu_parameters);

    free(p);
    free(l->parameters);
    free(l);
}

layer_vtable cuda_dense_vtable = {.forward = cuda_dense_forward, .free = free_cuda_dense_layer};

//TODO
layer* cnstr_cuda_dense_layer(const int inputCount, const int outputCount, const int threads, void (*initialize)(const layer* l)) {
    const auto l = (layer*)malloc(sizeof(layer));
    const auto p = (cuda_dense_layer_params*)malloc(sizeof(cuda_dense_layer_params));

    l->data = p;
    l->in_count = inputCount;
    l->out_count = outputCount;

    l->parameters_count = inputCount * outputCount + outputCount;
    l->parameters = (double*)malloc(sizeof(double) * l->parameters_count);

    l->initialize = initialize;
    l->vtable = &cuda_dense_vtable;

    cudaMalloc(&p->gpu_in, inputCount * sizeof(double));
    cudaMalloc(&p->gpu_out, outputCount * sizeof(double));
    cudaMalloc(&p->gpu_parameters, l->parameters_count * sizeof(double));

    p->threads = threads;
    p->blocks = (l->out_count + threads - 1) / threads;

    return l;
}