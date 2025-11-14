#ifndef LIST_H
#define LIST_H
#include <stdlib.h>

#define l_get(x, c, i) ((c*)x->data)[i]
#define l_pget(x, c, i) (((c*)x->data) + i)
#define l_set(x, c, i, v) ((c*)x->data)[i] = v
#define l_add(x, c, v) grow_if_needed(x); l_set(x, c, x->count++, v)
#define l_add_s_arr(x, c, v) for(int i = 0; i < v->count; i++) { \
        l_add(x, c, l_get(v, c, i)); \
    }

typedef struct list {
    void* data;
    size_t el_size;
    int capacity;
    int count;
} list;

typedef struct s_arr {
    void* data;
    int count;
} s_arr;

extern list* alloc_list(size_t el_size);
extern void free_list(list* l);
extern void grow_if_needed(list* list);

extern s_arr* alloc_s_arr(size_t el_size, int count);
extern void free_s_arr(s_arr* arr);

#endif //LIST_H
