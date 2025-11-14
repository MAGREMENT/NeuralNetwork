//
// Created by zacha on 11-10-25.
//

#ifndef YAML_H
#define YAML_H
#include "list.h"

typedef struct yaml_line {
    int indentation;
    char name[32];
    char value[32];
} yaml_line;

yaml_line constr_yl(int ind, char name[], char v[]);
yaml_line constr_d_yl(int ind, char name[], double v);
yaml_line constr_i_yl(int ind, char name[], int i);

void save_yaml(yaml_line* list, int count, const char* file);

#endif //YAML_H
