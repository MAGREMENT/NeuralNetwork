//
// Created by zacha on 23-10-25.
//

#ifndef BUILDER_H
#define BUILDER_H

#include "neural_network.h"
#include "Util/Collections/list.h"

typedef struct builder {
    list* list;
    int in_count;
    int cost_type;
    int thread_count;
    int optimizer;
} builder;

builder* alloc_builder();
void free_builder(builder* builder);

void b_dense(const builder* builder, int outputCount);
void b_activation(const builder* builder, int type, int outputCount);

neural_network* build(const builder* builder);

#endif //BUILDER_H
