//
// Created by zacha on 10-10-25.
//

#include "list.h"

#include <string.h>

#define LIST_BASE_CAPACITY 4

list* alloc_list(size_t el_size) {
    list* result = malloc(sizeof(list));
    result->el_size = el_size;
    result->count = 0;
    result->capacity = LIST_BASE_CAPACITY;
    result->data = malloc(el_size * LIST_BASE_CAPACITY);
    return result;
}

void free_list(list* l) {
    free(l->data);
    free(l);
}

list* grow_if_needed(list* list) {
    if (list->count < list->capacity) return list;

    list->capacity *= 2;
    list->data = realloc(list->data, list->capacity * list->el_size);

    return list;
}

s_arr* alloc_s_arr(size_t el_size, int count) {
    s_arr* result = malloc(sizeof(s_arr));
    result->count = count;
    result->data = malloc(el_size * count);
    return result;
}

void free_s_arr(s_arr* arr) {
    free(arr->data);
    free(arr);
}

int index_of_int(const list* l, const int v) {
    if (l->el_size != sizeof(int)) return -1;

    const int* values = l->data;
    for (int i = 0; i < l->count; i++) {
        if (values[i] == v) return i;
    }

    return -1;
}

inline int contains_int(const list* l, int v) {
    return index_of_int(l, v) + 1;
}

void remove_at(list* l, int ind) {
    memcpy((char*)l->data + ind * l->el_size, (char*)l->data + (ind + 1) * l->el_size, (l->count - ind - 1) * l->el_size);
    l->count--;
}