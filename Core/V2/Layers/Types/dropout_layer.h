//
// Created by zacha on 16-11-25.
//

#ifndef NEWIMPLEMENTATION_DROPOUT_LAYER_H
#define NEWIMPLEMENTATION_DROPOUT_LAYER_H

#include "../layer.h"

enum dropout_types {
    INVERTED,
    CLASSIC
};

layer* cnstr_dropout_layer(int type, int outCount, double rate);

#endif //NEWIMPLEMENTATION_DROPOUT_LAYER_H