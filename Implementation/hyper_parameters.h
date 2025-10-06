#ifndef HYPER_PARAMETERS_H
#define HYPER_PARAMETERS_H

#include "neural_network.h"

typedef struct yaml_hyper_parameter {
    char* name;
    char* value;
    int indentation;
} yaml_hyper_parameter;

void apply_default_hyper_params(neural_network* network);
void apply_hyper_params(neural_network* network, yaml_hyper_parameter* list, int count);

#endif //HYPER_PARAMETERS_H
