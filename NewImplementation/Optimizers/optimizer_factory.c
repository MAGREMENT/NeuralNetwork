//
// Created by zacha on 01-11-25.
//

#include "optimizer_factory.h"

#include <stddef.h>

#include "Types/adam_optimizer.h"
#include "Types/gradient_descent_optimizer.h"
#include "Types/momentum_gradient_descent_optimizer.h"
#include "Types/nesterov_optimizer.h"
#include "Types/rms_prop_optimizer.h"

inline optimizer* cnstr_optimizer(const int type, const optimizer_cnstr_args args) {
    switch (type) {
        case GRADIENT_DESCENT : return cnstr_gradient_descent_optimizer();
        case MOMENTUM_GRADIENT_DESCENT: return cnstr_momentum_gradient_descent_optimizer(args.value);
        case NESTEROV : return cnstr_nesterov_optimizer(args.value);
        case RMS_PROP : return cnstr_rms_prop_optimizer(args.value);
        case ADAM : return cnstr_adam_optimizer(args.values.v1, args.values.v2);
        default : return NULL;
    }
}
