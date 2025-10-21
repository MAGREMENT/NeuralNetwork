//
// Created by zacha on 21-10-25.
//

#include "logger.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

inline void flog(char format[], ...) {
#if LOG_ENABLED
    va_list args;
    va_start(args, format);
    const int size = vsnprintf(NULL, 0, format, args);
    va_end(args);

    if (size < 0) return;

    char* buffer = malloc((size + 1) * sizeof(char));
    va_start(args, format);
    vsnprintf(buffer, size + 1, format, args);
    va_end(args);

    FILE* fptr = fopen("log.txt", "a");
    fwrite(buffer, sizeof(char), size, fptr);
    fputc('\n', fptr);
    fclose(fptr);
    free(buffer);
#endif
}