//
// Created by zacha on 29-10-25.
//

#ifndef NEWIMPLEMENTATION_ADAM_OPTIMIZER_H
#define NEWIMPLEMENTATION_ADAM_OPTIMIZER_H

#include "../optimizer.h"
#include "../../multi-threading.h"

extern optimizer* cnstr_adam_optimizer(double beta1, double beta2);
optimizer* cnstr_mt_adam_optimizer(double beta1, double beta2, thread_pool* pool, int parallelCount);

#endif //NEWIMPLEMENTATION_ADAM_OPTIMIZER_H