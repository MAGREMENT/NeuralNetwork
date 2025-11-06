//
// Created by zacha on 09-10-25.
//

#ifndef MULTI_THREADING_H
#define MULTI_THREADING_H
#include <bemapiset.h>

typedef struct parallel_range {
    int from;
    int to;
} parallel_range;

typedef struct parallel_range_data {
    void* params;
    parallel_range range;
} parallel_range_data;

void* alloc_critical_section();
void enter_critical_section(void* section);
void exit_critical_section(void* section);
void free_critical_section(void* section);
void exec_range_parallel(LPTHREAD_START_ROUTINE func, void* params, int total, int threadCount);

#endif //MULTI_THREADING_H
