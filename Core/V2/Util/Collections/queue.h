//
// Created by zacha on 17-11-25.
//

#ifndef NEWIMPLEMENTATION_QUEUE_H
#define NEWIMPLEMENTATION_QUEUE_H
#include <stddef.h>

#define q_enq(q, c, d) ((c*)q->arr)[q->rear] = d; shift_rear(q)
#define q_deq(q, c) ((c*)shift_front(q))[0]

typedef struct queue {
    void* arr;
    size_t el_size;
    int capacity;
    int front;
    int rear;
} queue;

queue* alloc_queue(int initialCapacity, size_t el_size);
void free_queue(queue* queue);
extern int is_full(const queue* queue);
extern int is_empty(const queue* queue);
void shift_rear(queue* queue);
void* shift_front(queue* queue);

#endif //NEWIMPLEMENTATION_QUEUE_H