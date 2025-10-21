#ifndef HYPER_PARAMETERS_H
#define HYPER_PARAMETERS_H

#include "neural_network.h"
#include "yaml.h"

void apply_default_hyper_params(neural_network* network);
void apply_hyper_params(neural_network* network, yaml_line* list, int count);

list* alloc_get_hyper_params(neural_network* network);

#endif //HYPER_PARAMETERS_H
