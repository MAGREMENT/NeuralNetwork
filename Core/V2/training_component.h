//
// Created by zacha on 26-11-25.
//

#ifndef NEWIMPLEMENTATION_TRAINING_COMPONENT_H
#define NEWIMPLEMENTATION_TRAINING_COMPONENT_H

#define TCA_NONE (tc_cnstr_args) {.i = 0}
#define TCA_INT(v) (tc_cnstr_args) {.i = v}
#define TCA_DOUBLE(v) (tc_cnstr_args) {.d = v}
#define TCA_DOUBLE_INT(v1, v2) (tc_cnstr_args) {.di = (double_int) {v1, v2}}
#define TCA_DOUBLE2(v1, v2) (tc_cnstr_args) {.d2 = (double2) {v1, v2}}
#define TCA_DOUBLE3(v1, v2, v3) (tc_cnstr_args) {.d3 = (double3) {v1, v2, v3}}
#include "yaml.h"

typedef struct double_int {
    double d;
    int i;
} double_int;

typedef struct double2 {
    double d1;
    double d2;
} double2;

typedef struct double3 {
    double d1;
    double d2;
    double d3;
} double3;

typedef struct tc_cnstr_args {
    int i;
    double d;
    double_int di;
    double2 d2;
    double3 d3;
} tc_cnstr_args;

enum tc_cnstr_types {
    TCT_NONE,
    TCT_INT,
    TCT_DOUBLE,
    TCT_DOUBLE_INT,
    TCT_DOUBLE2,
    TCT_DOUBLE3,
};

typedef struct tc_metadata {
    char* name;
    int cnstr_type;
} tc_metadata;

void add_to_yaml_writer(yaml_writer* writer, tc_cnstr_args args, int args_type);
tc_cnstr_args get_args_from_yaml(yaml_reader* reader, int args_type);
int index_of_tc(const tc_metadata* arr, int count, const char* str, int def);

#endif //NEWIMPLEMENTATION_TRAINING_COMPONENT_H