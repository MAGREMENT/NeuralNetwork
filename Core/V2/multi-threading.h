//
// Created by zacha on 09-10-25.
//

#ifndef MULTI_THREADING_H
#define MULTI_THREADING_H

#include "Util/range.h"

typedef struct job_group job_group;
typedef struct thread_pool thread_pool;

typedef struct parallel_range_data {
    void* params;
    iteration_range range;
} parallel_range_data;

typedef struct parallel_range_executor {
    thread_pool* pool;
    job_group* group;
    parallel_range_data* data;
    int count;
} parallel_range_executor;

extern void init_critical_section(void* section);
extern void enter_critical_section(void* section);
extern void exit_critical_section(void* section);
extern void delete_critical_section(void* section);

extern job_group* alloc_job_group(int count);
extern void free_job_group(job_group* g);
extern void reset_group(job_group* g, int count);
extern void wait_for_group(job_group* g);

parallel_range_executor* alloc_pr_executor(thread_pool* pool, int count);
void free_pr_executor(parallel_range_executor* pre);

extern int get_processor_count();
extern int get_thread_count(thread_pool* pool);
thread_pool* alloc_thread_pool(int threadCount);
void free_thread_pool(thread_pool* pool);
void add_job(thread_pool* pool, unsigned long(*func)(void*), void* params, job_group* group);

extern void exec_parallel_range(const parallel_range_executor* executor, unsigned long(*func)(void*), void* params, range baseRange);

#endif //MULTI_THREADING_H
