//
// Created by zacha on 01-11-25.
//

#ifndef NEWIMPLEMENTATION_OPTIMIZER_FACTORY_H
#define NEWIMPLEMENTATION_OPTIMIZER_FACTORY_H
#include "optimizer.h"

enum optimizers {
    GRADIENT_DESCENT,
    MOMENTUM_GRADIENT_DESCENT,
    NESTEROV,
    RMS_PROP,
    ADAM
};

typedef struct double2 {
    double v1;
    double v2;
} double2;

typedef union optimizer_cnstr_args {
    double value;
    double2 values;
} optimizer_cnstr_args;

optimizer* cnstr_optimizer(int type, optimizer_cnstr_args args);

#endif //NEWIMPLEMENTATION_OPTIMIZER_FACTORY_H