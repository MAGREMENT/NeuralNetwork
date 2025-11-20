//
// Created by zacha on 09-10-25.
//

#include "multi-threading.h"

#include <windows.h>

#include "Util/Collections/queue.h"

inline void init_critical_section(void* section) {
    InitializeCriticalSection(section);
}

inline void enter_critical_section(void* section) {
    EnterCriticalSection(section);
}

inline void exit_critical_section(void* section) {
    LeaveCriticalSection(section);
}

inline void delete_critical_section(void* section) {
    DeleteCriticalSection(section);
}

struct job_group {
    long counter;
    HANDLE event;
};

inline job_group* alloc_job_group(const int count) {
    job_group* g = malloc(sizeof(job_group));
    *g = (job_group) {count, CreateEvent(NULL, FALSE, FALSE, NULL)};
    return g;
}

inline void free_job_group(job_group* g) {
    CloseHandle(g->event);
    free(g);
}

inline void reset_group(job_group* g, const int count) {
    //No need to reset the event Handle, since it's in auto-reset mode
    g->counter = count;
}

inline void wait_for_group(job_group* g) {
    WaitForSingleObject(g->event, INFINITE);
}

enum thread_job_types {
    EXEC,
    STOP
};

typedef struct thread_job {
    int type;
    unsigned long (*func)(void*);
    void* params;
    job_group* group;
} thread_job;

typedef struct thread_pool {
    queue* jobs;
    HANDLE* threads;
    int count;

    CRITICAL_SECTION cs;
    CONDITION_VARIABLE cv;
    int stopping;
} thread_pool;

static thread_job take_job(thread_pool* pool) {
    EnterCriticalSection(&pool->cs);

    while (is_empty(pool->jobs) && !pool->stopping) {
        SleepConditionVariableCS(&pool->cv, &pool->cs, INFINITE);
    }

    const thread_job job = pool->stopping ? (thread_job){.type = STOP} : q_deq(pool->jobs, thread_job);

    LeaveCriticalSection(&pool->cs);
    return job;
}

static unsigned long exec(void* params) {
    thread_pool* tp = params;
    while (1) {
        const thread_job job = take_job(tp);
        if (job.type == STOP) break;

        job.func(job.params);
        if (job.group != NULL) {
            if (InterlockedDecrement(&job.group->counter) == 0) {
                SetEvent(job.group->event);
            }
        }
    }

    return 0;
}

thread_pool* alloc_thread_pool(const int threadCount) {
    thread_pool* pool = malloc(sizeof(thread_pool));

    pool->jobs = alloc_queue(threadCount, sizeof(thread_job));
    pool->threads = malloc(sizeof(HANDLE) * threadCount);
    pool->count = threadCount;
    pool->stopping = 0;

    InitializeCriticalSection(&pool->cs);
    InitializeConditionVariable(&pool->cv);

    for (int i = 0; i < threadCount; i++) {
        pool->threads[i] = CreateThread(
            NULL,
            0,
            exec,
            pool,
            0,
            NULL);
    }

    return pool;
}

void free_thread_pool(thread_pool* pool) {
    pool->stopping = 1;

    WakeAllConditionVariable(&pool->cv);

    WaitForMultipleObjects(pool->count, pool->threads, TRUE, INFINITE);

    for (int i = 0; i < pool->count; i++) {
        CloseHandle(pool->threads[i]);
    }

    DeleteCriticalSection(&pool->cs);
    free(pool->threads);
    free_queue(pool->jobs);
    free(pool);
}

void add_job(thread_pool* pool, unsigned long (*func)(void*), void* params, job_group* group) {
    EnterCriticalSection(&pool->cs);
    const thread_job job = {EXEC, func, params, group};
    q_enq(pool->jobs, thread_job, job);
    LeaveCriticalSection(&pool->cs);

    WakeConditionVariable(&pool->cv);
}

void exec_range_parallel(unsigned long(* func)(void *), void* params, const int total, const int threadCount) {
    HANDLE* threads = malloc(sizeof(HANDLE) * threadCount);
    parallel_range_data* data = malloc(sizeof(parallel_range_data) * threadCount);
    range_split_iterator it = range_split(total, threadCount);

    for (int i = 0; i < threadCount; i++) {
        parallel_range_data* pd = data + i;

        pd->params = params;
        pd->range = range_split_next(&it);

        threads[i] = CreateThread(
            NULL,
            0,
            func,
            pd,
            0,
            NULL);
    }

    WaitForMultipleObjects(threadCount, threads, TRUE, INFINITE);

    for (int i = 0; i < threadCount; i++) {
        CloseHandle(threads[i]);
    }

    free(threads);
    free(data);
}

void exec_range_parallel_worker(unsigned long (*func)(void*), void* params, const int total, const worker_context* context) {
    parallel_range_data* data = malloc(sizeof(parallel_range_data) * context->pool->count);
    range_split_iterator it = range_split(total, context->pool->count);

    reset_group(context->group, context->pool->count);

    for (int i = 0; i < context->pool->count; i++) {
        parallel_range_data* pd = data + i;

        pd->params = params;
        pd->range = range_split_next(&it);

        add_job(context->pool, func, pd, context->group);
    }

    WaitForSingleObject(context->group->event, INFINITE);
    free(data);
}