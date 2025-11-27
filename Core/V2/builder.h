//
// Created by zacha on 23-10-25.
//

#ifndef BUILDER_H
#define BUILDER_H

#include "neural_network.h"
#include "training_component.h"
#include "yaml.h"

#include "Util/size.h"
#include "Util/Collections/list.h"

typedef struct builder_params {
    int batch_threads;
    int dense_mt_threshold;
    int optimizer_mt_threshold;
    int (*get_mt_count)(int operationCount, int total);
} builder_params;

typedef struct builder {
    list* list;
    size3D in_size;

    double learningRate;
    int shuffleDataOnIteration;

    int cost_type;

    int optimizer;
    tc_cnstr_args opt_args;

    int scheduler;
    tc_cnstr_args sch_args;

    int data_selector;
    tc_cnstr_args ds_args;
} builder;

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

extern builder_params def_b_params();
extern builder_params st_b_params();

extern builder* alloc_builder(int inSize);
extern builder* alloc_builder_3D(size3D inSize);
extern void free_builder(builder* builder);

void from_yaml(builder* builder, yaml_reader* r);
void to_yaml(const builder* builder, yaml_writer* w);

extern void b_opt(builder* builder, int type, tc_cnstr_args args);
extern void b_sch(builder* builder, int type, tc_cnstr_args args);
extern void b_ds(builder* builder, int type, tc_cnstr_args args);

extern void b_dense(const builder* builder, int outputCount);
extern void b_activation(const builder* builder, int type);
extern void b_conv(const builder* builder, size2D kernelSize, int kernelCount, int stride, int padding);
extern void b_pooling(const builder* builder, int type, size2D windowSize, int stride, int padding);

extern neural_network* build(const builder* builder, builder_params params);
extern neural_network* build_free(builder* builder, builder_params params);

#endif //BUILDER_H
