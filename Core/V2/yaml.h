//
// Created by zacha on 11-10-25.
//

#ifndef YAML_H
#define YAML_H

#define YAML_LINE_MAX_LENGTH 32

typedef struct yaml_line {
    int indentation;
    char name[YAML_LINE_MAX_LENGTH];
    char value[YAML_LINE_MAX_LENGTH];
} yaml_line;

extern yaml_line cnstr_yl(int ind, char name[], char* v);
extern yaml_line cnstr_d_yl(int ind, char name[], double v);
extern yaml_line cnstr_i_yl(int ind, char name[], int i);

void save_yaml(yaml_line* list, int count, const char* file);

#endif //YAML_H
