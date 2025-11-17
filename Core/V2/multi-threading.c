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

enum thread_job_types {
    EXEC,
    STOP
};

typedef struct thread_job {
    int type;
    void (*func)(void *);
    void* params;
} thread_job;

typedef struct windows_tp {
    queue* jobs;
    HANDLE* threads;
    int count;

    LPCRITICAL_SECTION cs;
    PCONDITION_VARIABLE cv;
    int stopping;
} windows_tp;

static thread_job take_job(thread_pool* pool) {
    const windows_tp* w_pool = pool;

    EnterCriticalSection(w_pool->cs);

    while (is_empty(w_pool->jobs) && !w_pool->stopping) {
        SleepConditionVariableCS(w_pool->cv, w_pool->cs, INFINITE);
    }

    const thread_job job = w_pool->stopping ? (thread_job){.type = STOP} : q_deq(w_pool->jobs, thread_job);

    LeaveCriticalSection(w_pool->cs);
    return job;
}

static unsigned long windows_tp_exec(void* params) {
    windows_tp* tp = params;
    while (1) {
        const thread_job job = take_job(tp);
        if (job.type == STOP) break;
        job.func(job.params);
    }

    return 0;
}

thread_pool* alloc_thread_pool(const int threadCount) {
    windows_tp* pool = malloc(sizeof(windows_tp));

    pool->jobs = alloc_queue(threadCount, sizeof(thread_job));
    pool->threads = malloc(sizeof(HANDLE) * threadCount);
    pool->count = threadCount;

    InitializeCriticalSection(pool->cs);
    InitializeConditionVariable(pool->cv);

    for (int i = 0; i < threadCount; i++) {
        pool->threads[i] = CreateThread(
            NULL,
            0,
            windows_tp_exec,
            pool,
            0,
            NULL);
    }

    return pool;
}

void free_thread_pool(thread_pool* pool) {
    windows_tp* w_pool = pool;

    w_pool->stopping = 1;

    WakeAllConditionVariable(w_pool->cv);

    WaitForMultipleObjects(w_pool->count, w_pool->threads, TRUE, INFINITE);

    for (int i = 0; i < w_pool->count; i++) {
        CloseHandle(w_pool->threads[i]);
    }

    DeleteCriticalSection(w_pool->cs);
    free(w_pool->threads);
    free_queue(w_pool->jobs);
    free(w_pool);
}

void add_job(thread_pool* pool, void(*func)(void*), void* params) {
    const windows_tp* w_pool = pool;

    EnterCriticalSection(w_pool->cs);
    const thread_job job = {EXEC, func, params};
    q_enq(w_pool->jobs, thread_job, job);
    LeaveCriticalSection(w_pool->cs);

    WakeConditionVariable(w_pool->cv);
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

void exec_range_parallel_th(void (*func)(void*), void* params, int total, thread_pool* pool) {
    windows_tp* w_pool = pool;
    parallel_range_data* data = malloc(sizeof(parallel_range_data) * w_pool->count);
    range_split_iterator it = range_split(total, w_pool->count);

    for (int i = 0; i < w_pool->count; i++) {
        parallel_range_data* pd = data + i;

        pd->params = params;
        pd->range = range_split_next(&it);

        add_job(pool, func, pd);
    }

    free(data);
}