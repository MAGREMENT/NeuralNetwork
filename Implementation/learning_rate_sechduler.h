//
// Created by zacha on 01-10-25.
//

#ifndef LEARNING_RATE_SECHDULING_H
#define LEARNING_RATE_SECHDULING_H

typedef struct learning_rate_scheduler learning_rate_scheduler;

struct learning_rate_scheduler {
    void* params;
    double (*schedule)(learning_rate_scheduler* sch, double learningRate, int iteration);
    void (*free)(learning_rate_scheduler* sch);
};

learning_rate_scheduler* constr_constant_scheduler();
learning_rate_scheduler* constr_iteration_decay_scheduler(double decay);
learning_rate_scheduler* constr_exponential_decay_scheduler(double decay);
learning_rate_scheduler* constr_inverse_decay_scheduler(double decay);
learning_rate_scheduler* constr_cosine_decay_scheduler(double endingLr, int iterationSpan);
learning_rate_scheduler* constr_varying_scheduler(); //TODO

#endif //LEARNING_RATE_SECHDULING_H
