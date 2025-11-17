//
// Created by zacha on 09-10-25.
//

#ifndef MULTI_THREADING_H
#define MULTI_THREADING_H

#include "Util/range.h"

typedef struct parallel_range_data {
    void* params;
    range range;
} parallel_range_data;

typedef void thread_pool;

extern void init_critical_section(void* section);
extern void enter_critical_section(void* section);
extern void exit_critical_section(void* section);
extern void delete_critical_section(void* section);

thread_pool* alloc_thread_pool(int threadCount);
void free_thread_pool(thread_pool* pool);
void add_job(thread_pool* pool, void(*func)(void*), void* params);

extern void exec_range_parallel(unsigned long(* func)(void *), void* params, int total, int threadCount);
extern void exec_range_parallel_th(void (*func)(void*), void* params, int total, thread_pool* pool);

#endif //MULTI_THREADING_H
