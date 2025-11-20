//
// Created by zacha on 19-11-25.
//

#include "tester.h"

#include <stdio.h>

#include "Util/console.h"

inline void fail_curr_test(const test_context* context, char* msg) {
    context->results[context->curr] = msg;
}

void run_tests(test_context* context) {
    enable_ANSI_color_codes();
    context->results = calloc(context->cases->count, sizeof(char*));

    for (context->curr = 0; context->curr < context->cases->count; context->curr++) {
        const test_case c = l_get(context->cases, test_case, context->curr);
        printf("#%d : %s ... ", context->curr + 1, c.name);
        c.func(context);

        char* result = context->results[context->curr];
        if (result == NULL) printf("\033[32mOK !\033[0m\n");
        else printf("\033[31mFAIL ! %s\033[0m\n", result);
    }

    free(context->results);
    free_list(context->cases);
}