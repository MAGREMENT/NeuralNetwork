//
// Created by zacha on 27-10-25.
//

#ifndef SCHEDULER_H
#define SCHEDULER_H

typedef struct scheduler scheduler;

typedef struct scheduler_vtable {
    double (*schedule)(const scheduler* sch, double learningRate, int iteration);
    void (*free)(scheduler* sch);
} scheduler_vtable;

struct scheduler {
    void* params;
    scheduler_vtable* vtable;
};

void free_empty_scheduler(scheduler* sch);
void free_def_scheduler(scheduler* sch);

#endif //SCHEDULER_H