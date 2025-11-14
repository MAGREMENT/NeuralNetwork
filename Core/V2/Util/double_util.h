//
// Created by zacha on 02-11-25.
//

#ifndef NEWIMPLEMENTATION_DOUBLE_UTIL_H
#define NEWIMPLEMENTATION_DOUBLE_UTIL_H

extern int d_max_ind(const double* values, int count);

//Double equals
extern int deq(double left, double right, double margin);

//Default double equals
extern int def_deq(double left, double right);

extern char* alloc_dseq_to_str(const double* values, int count);

#endif //NEWIMPLEMENTATION_DOUBLE_UTIL_H