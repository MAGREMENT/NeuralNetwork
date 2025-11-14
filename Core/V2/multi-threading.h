//
// Created by zacha on 09-10-25.
//

#ifndef MULTI_THREADING_H
#define MULTI_THREADING_H

typedef struct parallel_range {
    int from;
    int to;
} parallel_range;

typedef struct parallel_range_data {
    void* params;
    parallel_range range;
} parallel_range_data;

extern void* alloc_critical_section();
extern void enter_critical_section(void* section);
extern void exit_critical_section(void* section);
extern void free_critical_section(void* section);
extern void exec_range_parallel(unsigned long(* func)(void *), void* params, int total, int threadCount);

#endif //MULTI_THREADING_H
