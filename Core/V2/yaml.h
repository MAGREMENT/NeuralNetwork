//
// Created by zacha on 11-10-25.
//

#ifndef YAML_H
#define YAML_H

#define YAML_LINE_MAX_LENGTH 32
#include "Util/Collections/list.h"

typedef struct yaml_line {
    int indentation;
    int is_array;
    char name[YAML_LINE_MAX_LENGTH];
    char value[YAML_LINE_MAX_LENGTH];
} yaml_line;

typedef struct yaml_writer {
    list* lines;
    int indentation;
    list* array_indentations;
} yaml_writer;

extern yaml_line cnstr_yl(int ind, int is_array, char name[], char* v);
extern yaml_line cnstr_d_yl(int ind, int is_array, char name[], double v);
extern yaml_line cnstr_i_yl(int ind, int is_array, char name[], int i);

extern yaml_writer* alloc_yaml_writer();
extern void free_yaml_writer(yaml_writer* writer);

extern void yw_begin_map(yaml_writer* writer);
extern void yw_begin_arr(yaml_writer* writer);
extern void yw_end(yaml_writer* writer);

extern void yw_str_n(yaml_writer* writer, char name[]);
extern void yw_str_nv(yaml_writer* writer, char name[], char value[]);
extern void yw_int_nv(yaml_writer* writer, char name[], int i);
extern void yw_int_v(yaml_writer* writer, int i);
extern void yw_d_nv(yaml_writer* writer, char name[], double b);

void save_yaml(yaml_writer* writer, const char* file);

#endif //YAML_H
