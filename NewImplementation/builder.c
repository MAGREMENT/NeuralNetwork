//
// Created by zacha on 23-10-25.
//

#include "builder.h"

#include "asserter.h"
#include "Layers/Types/activation_layer.h"
#include "Layers/Types/dense_layer.h"

typedef struct dense_element {
    int out_count;
} dense_element;

typedef struct activation_element {
    int type;
    int out_count;
} activation_element;

typedef union builder_union {
    dense_element dense;
    activation_element activation;
} builder_union ;

typedef struct builder_element {
    int type;
    builder_union element;
} builder_element;

builder* alloc_builder() {
    builder* b = malloc(sizeof(builder));
    b->list = alloc_list(sizeof(builder_element));

    b->in_count = -1;

    b->thread_count = 1;

    b->cost_type = -1;
    b->optimizer = -1;

    return b;
}

void free_builder(builder* builder) {
    free(builder->list);
    free(builder);
}

void b_dense(const builder* builder, const int outputCount) {
    builder_element el;
    el.type = DENSE;
    el.element.dense.out_count = outputCount;
    l_add(builder->list, builder_element, el);
}

void b_activation(const builder* builder, const int type, const int outputCount) {
    builder_element el;
    el.type = ACTIVATION;
    el.element.activation.type = type;
    el.element.activation.out_count = outputCount;
    l_add(builder->list, builder_element, el);
}

neural_network* build(const builder* builder) {
    if (builder->in_count <= 0) return NULL;

    neural_network* n = alloc_neural_network(builder->list->count);

    n->threadCount = builder->thread_count;
    n->cost_vtable = cost_vtables + builder->cost_type;
    //TODO optimizers

    int in_count = builder->in_count;
    for (int i = 0; i < builder->list->count; i++) {
        const builder_element el = l_get(builder->list, builder_element, i);
        switch (el.type) {
            case DENSE :
                const dense_element de = el.element.dense;

                void (*initialize)(const layer*) = initialize_dense_random;
                for (int j = i + 1; j < builder->list->count; j++) {
                    const builder_element buffer = l_get(builder->list, builder_element, i);
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
                }

                n->layers[i] = cnstr_dense_layer(in_count, de.out_count, initialize);
                in_count = de.out_count;

                break;
            case ACTIVATION :
                const activation_element ae = el.element.activation;
                n->layers[i] = cnstr_activation_layer(ae.type, ae.out_count);

                break;
            default:
                assert(false); //Should not happen
                free_neural_network(n, true);
                return NULL;
        }
    }

    return n;
}


