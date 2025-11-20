//
// Created by zacha on 23-10-25.
//

#include "builder.h"

#include "asserter.h"
#include "neural_network.h"
#include "Layers/Types/activation_layer.h"
#include "Layers/Types/convolutional_layer.h"
#include "Layers/Types/dense_layer.h"
#include "Layers/Types/pooling_layer.h"

#ifdef _MSC_VER
#include "Layers/Types/Cuda/cuda_dense_layer.cuh"
#endif

typedef struct dense_element {
    int out_count;
} dense_element;

typedef struct activation_element {
    int type;
} activation_element;

typedef struct conv_element {
    size2D kernel_size;
    int kernel_count;
    int stride;
    int padding;
} conv_element;

typedef struct pooling_element {
    int type;
    size2D window_size;
    int stride;
    int padding;
} pooling_element;

typedef union builder_union {
    dense_element dense;
    activation_element activation;
    conv_element conv;
    pooling_element pooling;
} builder_union ;

typedef struct builder_element {
    int type;
    builder_union element;
} builder_element;

builder_params def_b_params() {
    return (builder_params) {
        100 * 100,
        4,
        500 * 500,
        256
    };
}

extern builder_params st_b_params() {
    return (builder_params) {
        .dense_mt_threshold = INT_MAX,
        .dense_gpu_threshold = INT_MAX
    };
}

inline builder* alloc_builder(int inSize) {
    return alloc_builder_3D((size3D){inSize, 0, 0});
}

builder* alloc_builder_3D(const size3D inSize) {
    builder* b = malloc(sizeof(builder));
    b->list = alloc_list(sizeof(builder_element));

    b->in_size = inSize;

    b->learningRate = 1;
    b->shuffleDataOnIteration = 0;

    b->cost_type = -1;
    b->optimizer = -1;
    b->scheduler = -1;
    b->data_selector = -1;

    return b;
}

inline void free_builder(builder* builder) {
    free(builder->list);
    free(builder);
}

builder* from_yaml(const yaml_line* list, int count) {
    return NULL; //TODO
}

void to_yaml(const builder* builder, list* list) {
    int indentation = 0;

    l_add(list, yaml_line, cnstr_yl(indentation, "builder", ""));
    indentation++;

    //TODO
}

inline void b_opt(builder* builder, const int type, const optimizer_cnstr_args args) {
    builder->optimizer = type;
    builder->opt_args = args;
}

inline void b_sch(builder* builder, const int type, const scheduler_cnstr_args args) {
    builder->scheduler = type;
    builder->sch_args = args;
}

inline void b_ds(builder* builder, const int type, const data_selector_cnstr_args args) {
    builder->data_selector = type;
    builder->ds_args = args;
}

inline void b_dense(const builder* builder, const int outputCount) {
    builder_element el;
    el.type = DENSE;
    el.element.dense.out_count = outputCount;
    l_add(builder->list, builder_element, el);
}

inline void b_activation(const builder* builder, const int type) {
    builder_element el;
    el.type = ACTIVATION;
    el.element.activation.type = type;
    l_add(builder->list, builder_element, el);
}

inline void b_conv(const builder* builder, size2D kernelSize, int kernelCount, int stride, int padding) {
    builder_element el;
    el.type = CONVOLUTIONAL;
    el.element.conv.kernel_size = kernelSize;
    el.element.conv.kernel_count = kernelCount;
    el.element.conv.stride = stride;
    el.element.conv.padding = padding;
    l_add(builder->list, builder_element, el);
}
inline void b_pooling(const builder* builder, int type, size2D windowSize, int stride, int padding) {
    builder_element el;
    el.type = POOLING;
    el.element.pooling.type = type;
    el.element.pooling.window_size = windowSize;
    el.element.pooling.stride = stride;
    el.element.pooling.padding = padding;
    l_add(builder->list, builder_element, el);
}

static void* get_initialize(const builder* builder, const int i, void (*def)(const layer*), void (*xavier)(const layer*), void (*he)(const layer*)) {
    void (*initialize)(const layer*) = def;
    for (int j = i + 1; j < builder->list->count; j++) {
        const builder_element buffer = l_get(builder->list, builder_element, j);
        if (buffer.type != ACTIVATION) continue;

        const activation_element next = buffer.element.activation;
        switch (next.type) {
            case SIGMOID : case TANH :
                initialize = xavier;
                break;
            case RELU : case LEAKY_RELU :
                initialize = he;
                break;
            default :
                break;
        }

        break;
    }

    return initialize;
}

static int get_total_size(const size3D size) {
    int total = 1;
    if (size.width > 0) total *= size.width;
    if (size.height > 0) total *= size.height;
    if (size.depth > 0) total *= size.depth;

    return total;
}

static void free_wc_params(neural_network* n) {
    worker_context* wc = n->params;
    free_thread_pool(wc->pool);
    free_job_group(wc->group);
    free(wc);
}

neural_network_vtable wc_table = {free_wc_params};

neural_network* build(const builder* builder, const builder_params params) {
    if (builder->in_size.width <= 0) return NULL;

    neural_network* n = alloc_neural_network(builder->list->count);

    n->learningRate = builder->learningRate;
    n->shuffleDataOnIteration = builder->shuffleDataOnIteration;

    if (builder->optimizer >= 0) n->optimizer = cnstr_optimizer(builder->optimizer, builder->opt_args);
    if (builder->data_selector >= 0) n->data_selector = cnstr_data_selector(builder->data_selector, builder->ds_args);
    if (builder->scheduler >= 0) n->scheduler = cnstr_scheduler(builder->scheduler, builder->sch_args);

    int softmax_bce_optimization = 0;
    int in;
    void (*initialize)(const layer*);

    worker_context* context = NULL;

    size3D inSize = builder->in_size;
    for (int i = 0; i < builder->list->count; i++) {
        const builder_element el = l_get(builder->list, builder_element, i);
        switch (el.type) {
            case DENSE :
                const dense_element de = el.element.dense;

                initialize = get_initialize(builder, i, initialize_dense_random, initialize_dense_xavier, initialize_dense_he);
                in = get_total_size(inSize);

                const int operationCount = in * de.out_count;

#ifdef _MSC_VER
                if (operationCount >= params.dense_gpu_threshold) {
                    n->layers[i] = cnstr_cuda_dense_layer(in, de.out_count, params.gpu_t_count, initialize);
                    goto d_end;
                }
#endif

                if (operationCount >= params.dense_mt_threshold) {
                    if (context == NULL) {
                        context = malloc(sizeof(worker_context));
                        context->pool = alloc_thread_pool(params.mt_t_count);
                        context->group = alloc_job_group(params.mt_t_count);

                        n->params = context;
                        n->vtable = &wc_table;
                    }
                    n->layers[i] = cnstr_worker_multi_thread_dense_layer(in, de.out_count, context, initialize);
                }
                else n->layers[i] = cnstr_dense_layer(in, de.out_count, initialize);

                d_end :

                inSize = (size3D){de.out_count, 0, 0};
                break;
            case ACTIVATION :
                const activation_element ae = el.element.activation;
                in = get_total_size(inSize);

                if (i == builder->list->count - 1 && ae.type == SOFTMAX && builder->cost_type == BINARY_CROSS_ENTROPY) {
                    softmax_bce_optimization = 1;
                    n->layers[i] = cnstr_softmax_bce_layer(in);
                } else n->layers[i] = cnstr_activation_layer(ae.type, in);

                break;
            case CONVOLUTIONAL :
                const conv_element ce = el.element.conv;
                initialize = get_initialize(builder, i, initialize_conv_random, initialize_conv_xavier, initialize_conv_he);

                layer* l = cnstr_conv_layer(inSize, ce.kernel_size, ce.kernel_count, ce.stride, ce.padding, initialize);
                n->layers[i] = l;
                inSize = ((conv_layer_params*)l->data)->output_size;

                break;
            case POOLING :
                const pooling_element pe = el.element.pooling;

                n->layers[i] = cnstr_pooling_layer(pe.type, inSize, pe.window_size, pe.stride, pe.padding);
                break;
            default:
                assert(0); //Should not happen
                free_neural_network(n, 1);
                return NULL;
        }
    }

    if (softmax_bce_optimization) n->cost_vtable = &softmax_bce_cost_vtable;
    else if (builder->cost_type >= 0) n->cost_vtable = cost_vtables + builder->cost_type;

    return n;
}

neural_network* build_free(builder* builder, const builder_params params) {
    neural_network* result = build(builder, params);
    free_builder(builder);
    return result;
}


