//
// Created by zacha on 28-10-25.
//

#ifndef NEWIMPLEMENTATION_COSINE_DECAY_SCHEDULER_H
#define NEWIMPLEMENTATION_COSINE_DECAY_SCHEDULER_H

#include "../scheduler.h"

scheduler* cnstr_cosine_decay_scheduler(double endingLearningRate, int iterationSpan);

#endif //NEWIMPLEMENTATION_COSINE_DECAY_SCHEDULER_H