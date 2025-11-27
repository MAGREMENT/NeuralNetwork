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

typedef struct yaml_reader {
    list* lines;
    int index;
    int indentation;
} yaml_reader;

extern yaml_line cnstr_empty_yl(int ind, int is_array);
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

extern yaml_reader* alloc_yaml_reader();
extern void free_yaml_reader(yaml_reader* reader);

extern int yr_seek(yaml_reader* reader, char name[]);
extern int yr_next(yaml_reader* reader);
extern int yr_enter(yaml_reader* reader);
extern int yr_exit(yaml_reader* reader);

extern int yr_int_v(yaml_reader* reader);
extern int yr_int_seekv(yaml_reader* reader, char name[], int def);
extern double yr_d_v(yaml_reader* reader);
extern double yr_d_seekv(yaml_reader* reader, char name[], double def);
extern void yr_str_n(yaml_reader* reader, char name[]);
extern void yr_str_v(yaml_reader* reader, char value[]);

void save_yaml(const yaml_writer* writer, const char* file);
void download_yaml(const yaml_reader* reader, const char* file);

#endif //YAML_H
