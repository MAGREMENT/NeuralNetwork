//
// Created by zacha on 09-10-25.
//

#ifndef MULTI_THREADING_H
#define MULTI_THREADING_H

typedef struct parallel_thread_info {
    int index;
    int total;
} parallel_thread_info;

void* alloc_critical_section();
void enter_critical_section(void* section);
void exit_critical_section(void* section);
void free_critical_section(void* section);
void exec_parallel(void(*func)(void* params, parallel_thread_info threadInfo), void* params, int threadCount);

#endif //MULTI_THREADING_H
