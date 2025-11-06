//
// Created by zacha on 09-10-25.
//

#include "multi-threading.h"
#include <windows.h>

inline void* alloc_critical_section() {
    CRITICAL_SECTION* cs = malloc(sizeof(CRITICAL_SECTION));
    InitializeCriticalSection(cs);
    return cs;
}

inline void enter_critical_section(void* section) {
    EnterCriticalSection(section);
}

inline void exit_critical_section(void* section) {
    LeaveCriticalSection(section);
}

inline void free_critical_section(void* section) {
    DeleteCriticalSection(section);
    free(section);
}

inline void exec_range_parallel(const LPTHREAD_START_ROUTINE func, void* params, const int total, const int threadCount) {
    HANDLE* threads = malloc(sizeof(HANDLE) * threadCount);
    parallel_range_data* data = malloc(sizeof(parallel_range_data) * threadCount);

    const int div = total / threadCount;
    const int add = total % threadCount;

    for (int i = 0; i < threadCount; i++) {
        parallel_range_data* pd = data + i;

        pd->params = params;
        pd->range.from = i * div;
        pd->range.to = (i + 1) * div + (i == threadCount - 1 ? add : 0);

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