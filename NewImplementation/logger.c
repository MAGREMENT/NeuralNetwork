//
// Created by zacha on 21-10-25.
//

#include "logger.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "Util/string_util.h"

char format_error[] = "ERROR WHILE FORMATING !";

inline void flog(char format[], ...) {
#if LOG_ENABLED
    va_list args;
    int size;

    va_start(args, format);
    char* buffer = alloc_format(format, args, &size);
    va_end(args);

    if (buffer == NULL) {
        buffer = format_error;
        size = sizeof(format_error);
    }

    FILE* fptr = fopen("log.txt", "a");

    fwrite(buffer, sizeof(char), size, fptr);
    fputc('\n', fptr);

    fclose(fptr);

    free(buffer);
#endif
}
