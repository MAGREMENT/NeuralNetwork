//
// Created by zacha on 09-10-25.
//

#include "multi-threading.h"

#include <bemapiset.h>
#include <windows.h>

void* alloc_critical_section() {
    CRITICAL_SECTION* cs = malloc(sizeof(CRITICAL_SECTION));
    InitializeCriticalSection(cs);
    return cs;
}

void enter_critical_section(void* section) {
    EnterCriticalSection(section);
}

void exit_critical_section(void* section) {
    LeaveCriticalSection(section);
}

void free_critical_section(void* section) {
    DeleteCriticalSection(section);
    free(section);
}

typedef struct parallel_handler {
    void (*func)(void* params, parallel_thread_info threadInfo);
    void* params;
    parallel_thread_info threadInfo;
} parallel_handler;

static DWORD exec_parallel_handler(LPVOID param) {
    parallel_handler* ph = param;
    ph->func(ph->params, ph->threadInfo);
    return 0;
}

void exec_parallel(void(*func)(void* params, parallel_thread_info threadInfo), void* params, int threadCount) {
    HANDLE* threads = malloc(sizeof(HANDLE) * threadCount);
    parallel_handler* phs = malloc(sizeof(parallel_handler) * threadCount);

    for (int i = 0; i < threadCount; i++) {
        parallel_handler* ph = ph + i;

        ph->func = func;
        ph->params = params;
        ph->threadInfo.index = i;
        ph->threadInfo.total = threadCount;

        threads[i] = CreateThread(
            NULL,
            0,
            exec_parallel_handler,
            ph,
            0,
            NULL);
    }

    WaitForMultipleObjects(threadCount, threads, TRUE, INFINITE);

    for (int i = 0; i < threadCount; i++) {
        CloseHandle(threads[i]);
    }

    free(threads);
    free(phs);
}