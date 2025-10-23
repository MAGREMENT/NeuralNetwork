#ifndef UTILS_H
#define UTILS_H

//TODO move to appropriate Util/X.c file

int max_index(double values[], int count);
//Double equals
int deq(double left, double right, double margin);
//Default double equals
int def_deq(double left, double right);
char* alloc_seq_to_str(double* values, int count);

#endif //UTILS_H
