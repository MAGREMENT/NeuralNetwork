//
// Created by zacha on 21-10-25.
//

#include "string_util.h"

#include <stdio.h>
#include <stdlib.h>

char* alloc_format(char format[], va_list args, int* size) {
    va_list copy;
    va_copy(copy, args);

    *size = vsnprintf(NULL, 0, format, args);

    if (*size < 0) {
        va_end(copy);
        return NULL;
    }

    char* buffer = malloc((*size + 1) * sizeof(char));
    vsnprintf(buffer, *size + 1, format, copy);

    va_end(copy);

    return buffer;
}
