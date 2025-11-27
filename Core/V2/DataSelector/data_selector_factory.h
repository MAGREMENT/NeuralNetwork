//
// Created by zacha on 01-11-25.
//

#ifndef NEWIMPLEMENTATION_DATA_SELECTOR_FACTORY_H
#define NEWIMPLEMENTATION_DATA_SELECTOR_FACTORY_H
#include "data_selector.h"
#include "../training_component.h"

#define DATA_SELECTOR_COUNT 2

enum data_selectors {
    FULL_BATCH,
    MINI_BATCH
};

extern tc_metadata ds_metadata[];

extern data_selector* cnstr_data_selector(int type, tc_cnstr_args args);

#endif //NEWIMPLEMENTATION_DATA_SELECTOR_FACTORY_H