//
// Created by zacha on 07-11-25.
//

#ifndef NEWIMPLEMENTATION_CUDA_DENSE_LAYER_CUH
#define NEWIMPLEMENTATION_CUDA_DENSE_LAYER_CUH

#include "../../layer.h"

#ifdef __cplusplus
extern "C" {
#endif

layer* cnstr_cuda_dense_layer(int inputCount, int outputCount, int threads, void (*initialize)(const layer* l));

#ifdef __cplusplus
}
#endif

#endif //NEWIMPLEMENTATION_CUDA_DENSE_LAYER_CUH