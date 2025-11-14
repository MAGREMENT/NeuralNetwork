//
// Created by zacha on 21-10-25.
//

#include "asserter.h"

#include <stdio.h>
#include <stdlib.h>

inline void assert(const int v) {
#if ASSERT_ENABLED
    if (!v) {
        printf("Assert failed");
        exit(EXIT_FAILURE);
    }
#endif
}

