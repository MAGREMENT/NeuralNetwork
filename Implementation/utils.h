#ifndef UTILS_H
#define UTILS_H


void init_random();
double rand_d(double min, double max);
int rand_i(int max);
double rand_std_nrml_distribution();
int max_index(double values[], int count);
//Double equals
int deq(double left, double right, double margin);
//Default double equals
int def_deq(double left, double right);
void list_remove(int* arr, int count, int index);
char* alloc_seq_to_str(double* values, int count);

#endif //UTILS_H
