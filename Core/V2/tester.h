//
// Created by zacha on 19-11-25.
//

#ifndef NEWIMPLEMENTATION_TESTER_H
#define NEWIMPLEMENTATION_TESTER_H

#include "Util/Collections/list.h"

#define CONTEXT test_context context = {.cases = alloc_list(sizeof(test_case))};
#define TEST(name) void name(test_context* context)
#define ADD_TEST(name) l_add(context.cases, test_case, ((test_case) {#name, name}));
#define TEARDOWN free:
#define ASSERT_M(b, m) do { \
    if(!(b)) { \
        fail_curr_test(context, m); \
        goto free;\
    } \
} while (0);
#define ASSERT(b) ASSERT_M(b, "")
#define RUN run_tests(&context);

typedef struct test_context test_context;

typedef struct test_case {
    const char* name;
    void (*func)(test_context*);
} test_case;

struct test_context {
    list* cases;
    char** results;
    int curr;
};

extern void fail_curr_test(const test_context* context, char* msg);
void run_tests(test_context* context);

#endif //NEWIMPLEMENTATION_TESTER_H