//
// Created by zacha on 01-11-25.
//

#include "optimizer_factory.h"

#include <stddef.h>

#include "Types/adadelta_optimizer.h"
#include "Types/adagrad_optimizer.h"
#include "Types/adam_optimizer.h"
#include "Types/simple_optimizer.h"
#include "Types/free_momentum_optimizer.h"
#include "Types/simplified_nesterov_optimizer.h"
#include "Types/proportional_momentum_optimizer.h"
#include "Types/rms_prop_optimizer.h"

inline optimizer* cnstr_optimizer(const int type, const optimizer_cnstr_args args) {
    switch (type) {
        case SIMPLE : return cnstr_simple_optimizer();
        case FREE_MOMENTUM: return cnstr_free_momentum_optimizer(args.value);
        case PROPORTIONAL_MOMENTUM : return cnstr_proportional_momentum_optimizer(args.value);
        case SIMPLIFIED_NESTEROV : return cnstr_simplified_nesterov_optimizer(args.value);
        case RMS_PROP : return cnstr_rms_prop_optimizer(args.value);
        case ADAM : return cnstr_adam_optimizer(args.values.v1, args.values.v2);
        case ADAGRAD : return cnstr_adagrad_optimizer();
        case ADADELTA : return cnstr_adadelta_optimizer(args.value);
        default : return NULL;
    }
}
