//
// Created by zacha on 16-11-25.
//

#include "dropout_layer.h"

#include <stdlib.h>

layer* cnstr_dropout_layer(const int type, const int outCount, const double rate) {
    layer* l = malloc(sizeof(layer));
    double* r = malloc(sizeof(double));
    *r = rate;

    l->data = r;
    l->in_count = outCount;
    l->out_count = outCount;

    l->parameters_count = 0;
    l->parameters = NULL;

    return l;
}
