//
// Created by zacha on 06-11-25.
//

#include "test.cuh"

__global__ void dense_f(const double *in, const double *w, const double* b, double *out, int out_count, int in_count, int n) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;

    if (o < n) {
		double v = b[o];

        for(int i = 0; i < in_count; i++){
            int ind = i * out_count + o;
            v += in[i] * w[ind];
        }

        out[o] = v;
	}
}