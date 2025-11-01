//
// Created by zacha on 01-11-25.
//

#ifndef NEWIMPLEMENTATION_DATA_SELECTOR_FACTORY_H
#define NEWIMPLEMENTATION_DATA_SELECTOR_FACTORY_H
#include "data_selector.h"

enum data_selectors {
    FULL_BATCH,
    MINI_BATCH
};

typedef union data_selector_cnstr_args {
    int value;
} data_selector_cnstr_args;

data_selector* cnstr_data_selector(int type, data_selector_cnstr_args args);

#endif //NEWIMPLEMENTATION_DATA_SELECTOR_FACTORY_H