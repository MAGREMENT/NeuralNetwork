//
// Created by zacha on 27-10-25.
//

#include "scheduler.h"

#include <stdlib.h>

inline void free_empty_scheduler(scheduler* sch) {
    free(sch);
}

inline void free_def_scheduler(scheduler* sch) {
    free(sch->params);
    free(sch);
}