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

typedef struct job_group job_group; //TODO threading pool specialized in range execution
typedef struct thread_pool thread_pool;

typedef struct worker_context {
    thread_pool* pool;
    job_group* group;
} worker_context;

extern void init_critical_section(void* section);
extern void enter_critical_section(void* section);
extern void exit_critical_section(void* section);
extern void delete_critical_section(void* section);

extern job_group* alloc_job_group(int count);
extern void free_job_group(job_group* g);
extern void reset_group(job_group* g, int count);
extern void wait_for_group(job_group* g);

thread_pool* alloc_thread_pool(int threadCount);
void free_thread_pool(thread_pool* pool);
void add_job(thread_pool* pool, unsigned long(*func)(void*), void* params, job_group* group);

extern void exec_range_parallel(unsigned long(* func)(void *), void* params, int total, int threadCount);
extern void exec_range_parallel_worker(unsigned long(*func)(void*), void* params, int total, const worker_context* context);

#endif //MULTI_THREADING_H
