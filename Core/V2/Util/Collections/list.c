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

void grow_if_needed(list* list) {
    if (list->count < list->capacity) return;

    list->capacity *= 2;
    void* buffer = malloc(list->el_size * list->capacity);
    memcpy(buffer, list->data, list->el_size * list->count);
    free(list->data);
    list->data = buffer;
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