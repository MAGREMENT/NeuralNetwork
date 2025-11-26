//
// Created by zacha on 11-10-25.
//

#include "yaml.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

inline yaml_line constr_yl(int ind, char name[], char v[]) {
    yaml_line p;
    p.indentation = ind;
    strcpy(p.name, name);
    strcpy(p.value, v);
    return p;
}

inline yaml_line constr_d_yl(int ind, char name[], double v) {
    yaml_line p;
    p.indentation = ind;
    strcpy(p.name, name);
    snprintf(p.value, sizeof(p.value), "%f", v);
    return p;
}

inline yaml_line constr_i_yl(int ind, char name[], int v) {
    yaml_line p;
    p.indentation = ind;
    strcpy(p.name, name);
    snprintf(p.value, sizeof(p.value), "%d", v);
    return p;
}

inline void save_yaml(yaml_line* list, int count, const char* file) {
    FILE* fptr = fopen(file, "w");

    for (int i = 0; i < count; i++) {
        yaml_line line = list[i];
        char* ind = malloc(sizeof(char) * (line.indentation * 2 + 1));
        int j = 0;
        for (; j < line.indentation * 2; j++) {
            ind[j] = ' ';
        }
        ind[j] = '\0';

        fprintf(fptr, "%s%s: %s\n", ind, line.name, line.value);
        free(ind);
    }

    fclose(fptr);
}