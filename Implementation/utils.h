#ifndef UTILS_H
#define UTILS_H


void init_random();
double random(double min, double max);
int rand_i(int max);
int max_index(double values[], int count);
//Double equals
int deq(double left, double right, double margin);
//Default double equals
int def_deq(double left, double right);
void list_remove(int* arr, int count, int index);

#endif //UTILS_H
