//
// Created by zacha on 17-11-25.
//

#include "queue.h"

#include <stdlib.h>
#include <string.h>

#include "../../asserter.h"

queue* alloc_queue(const int initialCapacity, const size_t el_size) {
    assert(initialCapacity > 0);

    queue* q = malloc(sizeof(queue));
    q->capacity = initialCapacity;
    q->el_size = el_size;
    q->arr = malloc(initialCapacity * el_size);
    q->front = 0;
    q->count  = 0;
    return q;
}

void free_queue(queue* queue) {
    free(queue->arr);
    free(queue);
}

inline int is_full(const queue* queue) {
    return queue->count >= queue->capacity;
}

inline int is_empty(const queue* queue) {
    return queue->count == 0;
}

void* shift_rear(queue* queue) {
    if (is_full(queue)) {
        queue->capacity *= 2;
        queue->arr = realloc(queue->arr, queue->capacity * queue->el_size);
        assert(queue->arr != NULL);

        if (queue->front != 0) {
            const int until = queue->front + queue->count - queue->capacity / 2;
            memcpy((char*)queue->arr + (queue->front + queue->count - 1) * queue->el_size, queue->arr, until * queue->el_size);
        }
    }

    const int e = (queue->front + queue->count) % queue->capacity;
    void* result = (char*)queue->arr + e * queue->el_size;
    queue->count++;

    return result;
}

void* shift_front(queue* queue) {
    assert(!is_empty(queue));

    void* result = (char*)queue->arr + queue->front * queue->el_size;
    queue->front = (queue->front + 1) % queue->capacity;
    queue->count--;
    return result;
}