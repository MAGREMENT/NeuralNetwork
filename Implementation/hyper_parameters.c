//
// Created by zacha on 06-10-25.
//

#include "hyper_parameters.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
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

void apply_hyper_params(neural_network* network, yaml_line* list, int count) {
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
        yaml_line curr = list[i];
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
    network->threadCount = tc;
    set_activation_type(network, a, oa);
    set_cost_type(network, c);
    set_optimizer(network, opt == NULL ? def_opt() : opt);
    set_data_selector(network, ds == NULL ? def_ds() : ds);
    set_scheduler(network, lrs == NULL ? def_lrs() : lrs);
}

inline list* alloc_get_hyper_params(neural_network* network) {
    list* result = alloc_list(sizeof(yaml_line));
    int indentation = 0;

    l_add(result, yaml_line, constr_yl(indentation, "neural_network", ""));
    indentation++;

    l_add(result, yaml_line, constr_d_yl(indentation, "learning_rate", network->learningRate));
    l_add(result, yaml_line, constr_i_yl(indentation, "shuffle_data_on_iteration", network->shuffleDataOnIteration));
    l_add(result, yaml_line, constr_i_yl(indentation, "thread_count", network->threadCount));

    int at, oat;
    get_activation_type(network, &at, &oat);
    l_add(result, yaml_line, constr_i_yl(indentation, "activation_type", at));
    l_add(result, yaml_line, constr_i_yl(indentation, "output_activation_type", oat));
    l_add(result, yaml_line, constr_i_yl(indentation, "cost_type", get_cost_type(network)));

    if (network->optimizer != NULL && network->optimizer->alloc_to_hyper != NULL) {
        s_arr* arr = network->optimizer->alloc_to_hyper(network->optimizer, indentation);
        l_add_s_arr(result, yaml_line, arr);
        free(arr);
    }

    if (network->scheduler != NULL && network->scheduler->alloc_to_hyper != NULL) {
        s_arr* arr = network->scheduler->alloc_to_hyper(network->scheduler, indentation);
        l_add_s_arr(result, yaml_line, arr);
        free(arr);
    }

    if (network->data_selector != NULL && network->data_selector->alloc_to_hyper != NULL) {
        s_arr* arr = network->data_selector->alloc_to_hyper(network->data_selector, indentation);
        l_add_s_arr(result, yaml_line, arr);
        free(arr);
    }

    return result;
}