//
// Created by zacha on 06-10-25.
//

#include "hyper_parameters.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

static int def_sdoi() {
    return false;
}

static double def_lr() {
    return 1;
}

static int def_tc() {
    return 1;
}

static int def_a() {
    return SIGMOID;
}

static int def_oa() {
    return SIGMOID;
}

static int def_c() {
    return MEAN_SQUARED;
}

static data_selector* def_ds() {
    return create_full_batch_selector();
}

static optimizer* def_opt() {
    return create_nesterov_optimizer(0.9);
}

static learning_rate_scheduler* def_lrs() {
    return constr_constant_scheduler();
}

inline void apply_default_hyper_params(neural_network* network) {
    network->shuffleDataOnIteration = def_sdoi();
    network->learningRate = def_lr();
    network->threadCount = def_tc();

    set_activation_type(network, def_a(), def_oa());
    set_cost_type(network, def_c());

    set_optimizer(network, def_opt());
    set_data_selector(network, def_ds());
    set_scheduler(network, def_lrs());
}

void apply_hyper_params(neural_network* network, yaml_hyper_parameter* list, int count) {
    int sdoi = def_sdoi();
    double lr = def_lr();
    int tc = def_tc();
    int a = def_a();
    int oa = def_oa();
    int c = def_c();
    data_selector* ds = NULL;
    optimizer* opt = NULL;
    learning_rate_scheduler* lrs = NULL;

    int i = 0;
    while (i < count) {
        yaml_hyper_parameter curr = list[i];
        /*switch (curr.name) {
            case "shuffleDataOnIteration" :
                sdoi = atoi(curr.value);
                break;
            //TODO continue
            default: break;
        }*/

        i += 1;
    }

    network->shuffleDataOnIteration = sdoi;
    network->learningRate = lr;
    network->threadCount = def_tc();
    set_activation_type(network, a, oa);
    set_cost_type(network, c);
    set_optimizer(network, opt == NULL ? def_opt() : opt);
    set_data_selector(network, ds == NULL ? def_ds() : ds);
    set_scheduler(network, lrs == NULL ? def_lrs() : lrs);
}
