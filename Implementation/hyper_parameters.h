#ifndef HYPER_PARAMETERS_H
#define HYPER_PARAMETERS_H

#include "old_nn.h"
#include "yaml.h"

void apply_default_hyper_params(old_nn* network);
void apply_hyper_params(old_nn* network, yaml_line* list, int count);

list* alloc_get_hyper_params(old_nn* network);

#endif //HYPER_PARAMETERS_H
