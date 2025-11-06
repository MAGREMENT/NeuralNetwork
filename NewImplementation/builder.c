//
// Created by zacha on 23-10-25.
//

#include "builder.h"

#include "asserter.h"
#include "neural_network.h"
#include "Layers/Types/activation_layer.h"
#include "Layers/Types/dense_layer.h"

typedef struct dense_element {
    int out_count;
} dense_element;

typedef struct activation_element {
    int type;
} activation_element;

typedef union builder_union {
    dense_element dense;
    activation_element activation;
} builder_union ;

typedef struct builder_element {
    int type;
    builder_union element;
} builder_element;

builder* alloc_builder(const int inCount) {
    builder* b = malloc(sizeof(builder));
    b->list = alloc_list(sizeof(builder_element));

    b->in_count = inCount;

    b->thread_count = 1;
    b->learningRate = 1;
    b->shuffleDataOnIteration = false;

    b->cost_type = -1;
    b->optimizer = -1;
    b->scheduler = -1;
    b->data_selector = -1;

    return b;
}

void free_builder(builder* builder) {
    free(builder->list);
    free(builder);
}

void b_opt(builder* builder, const int type, const optimizer_cnstr_args args) {
    builder->optimizer = type;
    builder->opt_args = args;
}

void b_sch(builder* builder, const int type, const scheduler_cnstr_args args) {
    builder->scheduler = type;
    builder->sch_args = args;
}

void b_ds(builder* builder, const int type, const data_selector_cnstr_args args) {
    builder->data_selector = type;
    builder->ds_args = args;
}

void b_dense(const builder* builder, const int outputCount) {
    builder_element el;
    el.type = DENSE;
    el.element.dense.out_count = outputCount;
    l_add(builder->list, builder_element, el);
}

void b_activation(const builder* builder, const int type) {
    builder_element el;
    el.type = ACTIVATION;
    el.element.activation.type = type;
    l_add(builder->list, builder_element, el);
}

static void* get_initialize(const builder* builder, const int i) {

    void (*initialize)(const layer*) = initialize_dense_random;
    for (int j = i + 1; j < builder->list->count; j++) {
        const builder_element buffer = l_get(builder->list, builder_element, j);
        if (buffer.type != ACTIVATION) continue;

        const activation_element next = buffer.element.activation;
        switch (next.type) {
            case SIGMOID : case TANH :
                initialize = initialize_dense_xavier;
                break;
            case RELU : case LEAKY_RELU :
                initialize = initialize_dense_he;
                break;
            default :
                break;
        }

        break;
    }

    return initialize;
}

neural_network* build(const builder* builder) {
    if (builder->in_count <= 0) return NULL;

    neural_network* n = alloc_neural_network(builder->list->count);

    n->threadCount = builder->thread_count;
    n->learningRate = builder->learningRate;
    n->shuffleDataOnIteration = builder->shuffleDataOnIteration;

    if (builder->optimizer >= 0) n->optimizer = cnstr_optimizer(builder->optimizer, builder->opt_args);
    if (builder->data_selector >= 0) n->data_selector = cnstr_data_selector(builder->data_selector, builder->ds_args);
    if (builder->scheduler >= 0) n->scheduler = cnstr_scheduler(builder->scheduler, builder->sch_args);

    bool softmax_bce_optimization = false;

    int in_count = builder->in_count;
    for (int i = 0; i < builder->list->count; i++) {
        const builder_element el = l_get(builder->list, builder_element, i);
        switch (el.type) {
            case DENSE :
                const dense_element de = el.element.dense;
                void (*initialize)(const layer*) = get_initialize(builder, i);

                n->layers[i] = cnstr_dense_layer(in_count, de.out_count, initialize);
                in_count = de.out_count;

                break;
            case ACTIVATION :
                const activation_element ae = el.element.activation;
                const int out = i == 0 ? builder->in_count - 1 : n->layers[i - 1]->out_count;

                if (i == builder->list->count - 1 && ae.type == SOFTMAX && builder->cost_type == BINARY_CROSS_ENTROPY) {
                    softmax_bce_optimization = true;
                    n->layers[i] = cnstr_softmax_bce_layer(out);
                } else n->layers[i] = cnstr_activation_layer(ae.type, out);

                break;
            default:
                assert(false); //Should not happen
                free_neural_network(n, true);
                return NULL;
        }
    }

    if (softmax_bce_optimization) n->cost_vtable = &softmax_bce_cost_vtable;
    else if (builder->cost_type >= 0) n->cost_vtable = cost_vtables + builder->cost_type;

    return n;
}

neural_network* build_free(builder* builder) {
    const auto result = build(builder);
    free_builder(builder);
    return result;
}


