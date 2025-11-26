//
// Created by zacha on 11-10-25.
//

#include "yaml.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

inline yaml_line cnstr_yl(const int ind, const int is_array, char name[], char* v) {
    yaml_line p;
    p.indentation = ind;
    p.is_array = is_array;
    strcpy_s(p.name, YAML_LINE_MAX_LENGTH, name);
    strcpy_s(p.value, YAML_LINE_MAX_LENGTH, v);
    return p;
}

inline yaml_line cnstr_d_yl(const int ind, const int is_array, char name[], double v) {
    yaml_line p;
    p.indentation = ind;
    p.is_array = is_array;
    strcpy_s(p.name, YAML_LINE_MAX_LENGTH, name);
    snprintf(p.value, sizeof(p.value), "%f", v);
    return p;
}

inline yaml_line cnstr_i_yl(const int ind, const int is_array, char name[], int v) {
    yaml_line p;
    p.indentation = ind;
    p.is_array = is_array;
    strcpy_s(p.name, YAML_LINE_MAX_LENGTH, name);
    snprintf(p.value, sizeof(p.value), "%d", v);
    return p;
}

inline yaml_writer* alloc_yaml_writer() {
    yaml_writer* writer = malloc(sizeof(yaml_writer));
    writer->lines = alloc_list(sizeof(yaml_line));
    writer->array_indentations = alloc_list(sizeof(int));
    writer->indentation = 0;
    return writer;
}
inline void free_yaml_writer(yaml_writer* writer) {
    free_list(writer->lines);
    free_list(writer->array_indentations);
    free(writer);
}

inline void yw_begin_map(yaml_writer* writer) {
    if (contains_int(writer->array_indentations, writer->indentation)) writer->indentation++;
    writer->indentation++;
}

inline void yw_begin_arr(yaml_writer* writer) {
    if (contains_int(writer->array_indentations, writer->indentation)) writer->indentation++;
    writer->indentation++;
    l_add(writer->array_indentations, int, writer->indentation);
}

inline void yw_end(yaml_writer* writer) {
    const int ind = index_of_int(writer->array_indentations, writer->indentation);
    if (ind >= 0) remove_at(writer->array_indentations, ind);
    writer->indentation--;
    if (contains_int(writer->array_indentations, writer->indentation - 1)) writer->indentation--;
}

inline void yw_str_n(yaml_writer* writer, char name[]) {
    l_add(writer->lines, yaml_line, cnstr_yl(writer->indentation,
        contains_int(writer->array_indentations, writer->indentation), name, ""));
}

inline void yw_str_nv(yaml_writer* writer, char name[], char value[]) {
    l_add(writer->lines, yaml_line, cnstr_yl(writer->indentation,
        contains_int(writer->array_indentations, writer->indentation), name, value));
}

inline void yw_int_nv(yaml_writer* writer, char name[], const int i) {
    l_add(writer->lines, yaml_line, cnstr_i_yl(writer->indentation,
        contains_int(writer->array_indentations, writer->indentation), name, i));
}

inline void yw_int_v(yaml_writer* writer, int i) {
    if (!contains_int(writer->array_indentations, writer->indentation)) return;

    l_add(writer->lines, yaml_line, cnstr_i_yl(writer->indentation,1, "", i));
}

inline void yw_d_nv(yaml_writer* writer, char name[], const double d) {
    l_add(writer->lines, yaml_line, cnstr_d_yl(writer->indentation,
        contains_int(writer->array_indentations, writer->indentation), name, d));
}

void save_yaml(yaml_writer* writer, const char* file) {
    FILE* fptr = fopen(file, "w");

    for (int i = 0; i < writer->lines->count; i++) {
        yaml_line line = l_get(writer->lines, yaml_line, i);
        char* ind = malloc(sizeof(char) * (line.indentation * 2 + 1));
        int j = 0;
        for (; j < line.indentation * 2; j++) {
            ind[j] = ' ';
        }
        ind[j] = '\0';

        fprintf(fptr, "%s%s%s%s%s\n", ind, line.is_array ? "- " : "", line.name, line.name[0] == '\0' ? "" : ": ", line.value);
        free(ind);
    }

    fclose(fptr);
}