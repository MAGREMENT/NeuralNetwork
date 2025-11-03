//
// Created by zacha on 02-11-25.
//

#ifndef NEWIMPLEMENTATION_DOUBLE_UTIL_H
#define NEWIMPLEMENTATION_DOUBLE_UTIL_H

int d_max_ind(const double* values, int count);

//Double equals
int deq(double left, double right, double margin);

//Default double equals
int def_deq(double left, double right);

char* alloc_dseq_to_str(const double* values, int count);

#endif //NEWIMPLEMENTATION_DOUBLE_UTIL_H