//
// Created by zacha on 23-10-25.
//

#ifndef BUILDER_H
#define BUILDER_H

#include "DataSelector/data_selector_factory.h"
#include "Optimizers/optimizer_factory.h"
#include "Schedulers/scheduler_factory.h"
#include "Util/Collections/list.h"

typedef struct builder {
    list* list;
    int in_count;

    int thread_count;
    double learningRate;
    bool shuffleDataOnIteration;

    int cost_type;

    int optimizer;
    optimizer_cnstr_args opt_args;

    int scheduler;
    scheduler_cnstr_args sch_args;

    int data_selector;
    data_selector_cnstr_args ds_args;
} builder;

builder* alloc_builder(int inCount);
void free_builder(builder* builder);

void b_opt(builder* builder, int type, optimizer_cnstr_args args);
void b_sch(builder* builder, int type, scheduler_cnstr_args args);
void b_ds(builder* builder, int type, data_selector_cnstr_args args);

void b_dense(const builder* builder, int outputCount);
void b_activation(const builder* builder, int type);

neural_network* build(const builder* builder);
neural_network* build_free(builder* builder);

#endif //BUILDER_H
