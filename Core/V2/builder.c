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

static int def_get_mt_count(const int operationCount, const int total) {
    if (operationCount >= 250 * 250) return total;
    return total / 2;
}

builder_params def_b_params() {
    return (builder_params) {
        4,
        100 * 100,
        100 * 100,
        def_get_mt_count,
    };
}

extern builder_params st_b_params() {
    return (builder_params) {
        .batch_threads = 1,
        .dense_mt_threshold = INT_MAX,
        .optimizer_mt_threshold = INT_MAX,
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

void to_yaml(const builder* builder, yaml_writer* w) {
    yw_str_n(w, "builder");
    yw_begin_map(w);

    yw_str_n(w, "input_size");
    yw_begin_arr(w);
    yw_int_v(w, builder->in_size.width);
    yw_int_v(w, builder->in_size.height);
    yw_int_v(w, builder->in_size.depth);
    yw_end(w);

    yw_str_n(w, "layers");
    yw_begin_arr(w);

    for (int i = 0; i < builder->list->count; i++) {
        const builder_element el = l_get(builder->list, builder_element, i);
        switch (el.type) {
            case DENSE :
                yw_str_n(w, "dense");
                yw_begin_map(w);

                yw_int_nv(w, "out_count", el.element.dense.out_count);
                yw_end(w);
                break;
            case ACTIVATION :
                yw_str_n(w, "activation");
                yw_begin_map(w);

                yw_str_nv(w, "type", activation_names[el.element.activation.type]);
                yw_end(w);
            default : break;
        }
    }

    yw_end(w);

    if (builder->optimizer >= 0) {
        yw_str_nv(w, "optimizer", opt_metadata[builder->optimizer].name);
        add_to_yaml_writer(w, builder->opt_args, opt_metadata[builder->optimizer].cnstr_type);
    }

    if (builder->scheduler >= 0) {
        yw_str_nv(w, "scheduler", sch_metadata[builder->scheduler].name);
        add_to_yaml_writer(w, builder->sch_args, sch_metadata[builder->scheduler].cnstr_type);
    }

    if (builder->data_selector >= 0) {
        yw_str_nv(w, "data_selector", ds_metadata[builder->data_selector].name);
        add_to_yaml_writer(w, builder->ds_args, ds_metadata[builder->data_selector].cnstr_type);
    }

    yw_int_nv(w, "cost_type", builder->cost_type);
    yw_int_nv(w, "shuffle_data_on_iteration", builder->shuffleDataOnIteration);
    yw_d_nv(w, "learning_rate", builder->learningRate);
}

inline void b_opt(builder* builder, const int type, const tc_cnstr_args args) {
    builder->optimizer = type;
    builder->opt_args = args;
}

inline void b_sch(builder* builder, const int type, const tc_cnstr_args args) {
    builder->scheduler = type;
    builder->sch_args = args;
}

inline void b_ds(builder* builder, const int type, const tc_cnstr_args args) {
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

inline void b_conv(const builder* builder, const size2D kernelSize, const int kernelCount, const int stride, const int padding) {
    builder_element el;
    el.type = CONVOLUTIONAL;
    el.element.conv.kernel_size = kernelSize;
    el.element.conv.kernel_count = kernelCount;
    el.element.conv.stride = stride;
    el.element.conv.padding = padding;
    l_add(builder->list, builder_element, el);
}
inline void b_pooling(const builder* builder, const int type, const size2D windowSize, const int stride, const int padding) {
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

neural_network* build(const builder* builder, const builder_params params) {
    if (builder->in_size.width <= 0) return NULL;

    neural_network* n = alloc_neural_network(builder->list->count);
    const int pool_treads = get_processor_count();
    int operation_threads = pool_treads;
    if (params.batch_threads > 1) {
        operation_threads = (operation_threads - params.batch_threads) / params.batch_threads;
        n->thread_pool = alloc_thread_pool(pool_treads);
        n->batch_executor = alloc_pr_executor(n->thread_pool, params.batch_threads);
    } else n->batch_executor = NULL;

    n->learningRate = builder->learningRate;
    n->shuffleDataOnIteration = builder->shuffleDataOnIteration;

    int softmax_bce_optimization = 0, in, max_param_count = 0;
    void (*initialize)(const layer*);

    size3D inSize = builder->in_size;
    for (int i = 0; i < builder->list->count; i++) {
        const builder_element el = l_get(builder->list, builder_element, i);
        switch (el.type) {
            case DENSE :
                const dense_element de = el.element.dense;

                initialize = get_initialize(builder, i, initialize_dense_random, initialize_dense_xavier, initialize_dense_he);
                in = get_total_size(inSize);

                const int operationCount = in * de.out_count;

                if (operationCount >= params.dense_mt_threshold && operation_threads > 1) {
                    if (n->thread_pool == NULL) n->thread_pool = alloc_thread_pool(pool_treads);
                    n->layers[i] = cnstr_multi_thread_dense_layer(in, de.out_count, n->thread_pool,
                        params.get_mt_count(operationCount, operation_threads), initialize);
                }
                else n->layers[i] = cnstr_dense_layer(in, de.out_count, initialize);
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

                n->layers[i] = cnstr_conv_layer(inSize, ce.kernel_size, ce.kernel_count, ce.stride, ce.padding, initialize);
                inSize = ((conv_layer_params*)n->layers[i]->data)->output_size;

                break;
            case POOLING :
                const pooling_element pe = el.element.pooling;

                n->layers[i] = cnstr_pooling_layer(pe.type, inSize, pe.window_size, pe.stride, pe.padding);
                inSize = ((pooling_layer_params*)n->layers[i]->data)->output_size;

                break;
            default:
                assert(0); //Should not happen
                free_neural_network(n, 1);
                return NULL;
        }

        max_param_count = n->layers[i]->parameters_count > max_param_count ? n->layers[i]->parameters_count : max_param_count;
    }

    if (softmax_bce_optimization) n->cost_vtable = &softmax_bce_cost_vtable;
    else if (builder->cost_type >= 0) n->cost_vtable = cost_vtables + builder->cost_type;

    if (builder->optimizer >= 0) {
        if (max_param_count >= params.optimizer_mt_threshold && pool_treads > 1) {
            if (n->thread_pool == NULL) n->thread_pool = alloc_thread_pool(pool_treads);
            n->optimizer = cnstr_mt_optimizer(builder->optimizer, builder->opt_args, n->thread_pool,
                def_get_mt_count(max_param_count,pool_treads));
        }
        else n->optimizer = cnstr_optimizer(builder->optimizer, builder->opt_args);
    }
    if (builder->data_selector >= 0) n->data_selector = cnstr_data_selector(builder->data_selector, builder->ds_args);
    if (builder->scheduler >= 0) n->scheduler = cnstr_scheduler(builder->scheduler, builder->sch_args);

    return n;
}

neural_network* build_free(builder* builder, const builder_params params) {
    neural_network* result = build(builder, params);
    free_builder(builder);
    return result;
}


