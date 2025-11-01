//
// Created by zacha on 01-11-25.
//

#include "data_selector_factory.h"

#include <stddef.h>

#include "Types/full_batch_data_selector.h"
#include "Types/mini_batch_data_selector.h"

data_selector* cnstr_data_selector(const int type, const data_selector_cnstr_args args) {
    switch (type) {
        case FULL_BATCH : return cnstr_full_batch_data_selector();
        case MINI_BATCH : return cnstr_mini_batch_data_selector(args.value);
        default: return NULL;
    }
}
