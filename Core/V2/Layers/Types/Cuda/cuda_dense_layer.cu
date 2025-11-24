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
    double* gpu_gradients;
    int threads;
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

__global__ void kernel_dense_backward(const double *d, const double *p, double *in, const int in_count, const int out_count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < in_count) {
        double v = 0;

        for(int o = 0; o < out_count; o++) {
            const double w = p[i * out_count + o];
            v += d[o] * w;
        }

        in[i] = v;
    }
}

__global__ void kernel_dense_delta_to_gradients(const double *in, const double* d, double *g, const int in_count, const int out_count) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;

    if (o < out_count) {
        for (int i = 0; i < in_count; i++) {
            g[i * out_count + o] += d[o] * in[i];
        }

        const int ind = out_count * in_count + o;
        g[ind] += d[o];
    }
}

static int get_blocks(const int count, const int threads) {
    return (count + threads - 1) / threads;
}

static void cuda_dense_forward(const layer* l, const double* inputs, double* outputs) {
    const auto p = (cuda_dense_layer_params*)l->data;
    cudaMemcpy(p->gpu_in, inputs, l->in_count * sizeof(double), cudaMemcpyHostToDevice);

    kernel_dense_forward<<<get_blocks(l->out_count, p->threads), p->threads>>>
        (p->gpu_in, p->gpu_parameters, p->gpu_out, l->in_count, l->out_count);

    cudaDeviceSynchronize();
    cudaMemcpy(outputs, p->gpu_out, l->out_count * sizeof(double), cudaMemcpyDeviceToHost);
}

static void cuda_dense_backward(const layer* l, const double* inputs, const double* deltas, double* outputs) {
    const auto p = (cuda_dense_layer_params*)l->data;
    cudaMemcpy(p->gpu_out, deltas, l->out_count * sizeof(double), cudaMemcpyHostToDevice);

    kernel_dense_backward<<<get_blocks(l->in_count, p->threads), p->threads>>>
          (p->gpu_out, p->gpu_parameters, p->gpu_in, l->in_count, l->out_count);

    cudaDeviceSynchronize();
    cudaMemcpy(outputs, p->gpu_in, l->in_count * sizeof(double), cudaMemcpyDeviceToHost);
}

static void cuda_dense_delta_to_gradients(const layer* l, const double* inputs, const double* deltas, double* gradients) {
    const auto p = (cuda_dense_layer_params*)l->data;
    cudaMemcpy(p->gpu_in, inputs, l->in_count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(p->gpu_out, deltas, l->out_count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(p->gpu_gradients, gradients, l->parameters_count * sizeof(double), cudaMemcpyHostToDevice);

    kernel_dense_delta_to_gradients<<<get_blocks(l->out_count, p->threads), p->threads>>>(
        p->gpu_in, p->gpu_out, p->gpu_gradients, l->in_count, l->out_count);

    cudaDeviceSynchronize();
    cudaMemcpy(gradients, p->gpu_gradients, l->parameters_count * sizeof(double), cudaMemcpyDeviceToHost);
}

static void free_cuda_dense_layer(layer* l) {
    const auto p = (cuda_dense_layer_params*)l->data;

    cudaFree(p->gpu_in);
    cudaFree(p->gpu_out);
    cudaFree(p->gpu_parameters);
    cudaFree(p->gpu_gradients);

    free(p);
    free(l->parameters);
    free(l);
}

layer_vtable cuda_dense_vtable = {NULL, cuda_dense_forward, NULL, cuda_dense_backward, cuda_dense_delta_to_gradients,
    free_cuda_dense_layer};

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
    cudaMalloc(&p->gpu_gradients, l->parameters_count * sizeof(double));

    p->threads = threads;

    return l;
}

inline void on_parameters_change(layer* l) {
    const auto p = (cuda_dense_layer_params*)l->data;
    cudaMemcpy(p->gpu_parameters, l->parameters, l->parameters_count * sizeof(double), cudaMemcpyHostToDevice);
}