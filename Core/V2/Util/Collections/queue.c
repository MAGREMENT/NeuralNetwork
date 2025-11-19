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
    q->rear = 0;
    return q;
}

void free_queue(queue* queue) {
    free(queue->arr);
    free(queue);
}

inline int is_full(const queue* queue) {
    if (queue->rear == queue->capacity - 1) {
        return queue->front == 0;
    }

    return queue->rear + 1 == queue->front;
}

inline int is_empty(const queue* queue) {
    return queue->front == queue->rear;
}

void shift_rear(queue* queue) {
    if (is_full(queue)) {
        const int old_c = queue->capacity;
        queue->capacity *= 2;
        void* buffer = malloc(queue->el_size * queue->capacity);

        if (queue->rear >= queue->front) {
            memcpy(buffer, queue->arr, old_c * queue->el_size);
        } else {
            const int temp = old_c - queue->front;
            memcpy(buffer, (char*)queue->arr + queue->front * queue->el_size, temp * queue->el_size);
            memcpy((char*)buffer + temp * queue->el_size, queue->arr, (queue->rear + 1) * queue->el_size);
        }

        free(queue->arr);
        queue->arr = buffer;
    }

    if (queue->rear == queue->capacity - 1) queue->rear = 0;
    else queue->rear++;
}

void* shift_front(queue* queue) {
    if (is_empty(queue)) return queue->arr;

    void* before = (char*)queue->arr + queue->front * queue->el_size;
    if (queue->front == queue->capacity - 1) queue->front = 0;
    else queue->front++;

    return before;
}