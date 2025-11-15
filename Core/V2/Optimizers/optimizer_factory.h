//
// Created by zacha on 01-11-25.
//

#ifndef NEWIMPLEMENTATION_OPTIMIZER_FACTORY_H
#define NEWIMPLEMENTATION_OPTIMIZER_FACTORY_H
#include "optimizer.h"

enum optimizers {
    SIMPLE,
    FREE_MOMENTUM,
    PROPORTIONAL_MOMENTUM,
    SIMPLIFIED_NESTEROV,
    RMS_PROP,
    ADAM,
    ADAGRAD,
    ADADELTA,
    ADAMAX,
    ADAMW
};

extern char* opt_names[];

typedef struct double2 {
    double v1;
    double v2;
} double2;

typedef struct double3 {
    double v1;
    double v2;
    double v3;
} double3;

typedef union optimizer_cnstr_args {
    double value;
    double2 value2;
    double3 value3;
} optimizer_cnstr_args;

extern optimizer* cnstr_optimizer(int type, optimizer_cnstr_args args);
extern char* get_opt_name(int type);

#endif //NEWIMPLEMENTATION_OPTIMIZER_FACTORY_H