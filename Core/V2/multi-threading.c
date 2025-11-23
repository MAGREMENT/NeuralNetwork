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

inline int get_processor_count() {
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
}

inline int get_thread_count(thread_pool* pool) {
    return pool->count;
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
    if (pool == NULL) return;

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

parallel_range_executor* alloc_pr_executor(thread_pool* pool, const int count) {
    parallel_range_executor* executor = malloc(sizeof(parallel_range_executor));
    executor->pool = pool;
    executor->group = alloc_job_group(count);
    executor->data = malloc(count * sizeof(parallel_range_data));
    executor->count = count;

    return executor;
}
void free_pr_executor(parallel_range_executor* pre) {
    if (pre == NULL) return;

    free_job_group(pre->group);
    free(pre->data);
    free(pre);
}

void exec_parallel_range(const parallel_range_executor* executor, unsigned long(*func)(void*), void* params, const range baseRange) {
    range_split_iterator it = range_split(baseRange.from, baseRange.to - baseRange.from, executor->count);
    reset_group(executor->group, executor->count);

    for (int i = 0; i < executor->count; i++) {
        parallel_range_data* pd = executor->data + i;

        pd->params = params;
        pd->range = to_iteration_range(range_split_next(&it), i);

        add_job(executor->pool, func, pd, executor->group);
    }

    wait_for_group(executor->group);
}