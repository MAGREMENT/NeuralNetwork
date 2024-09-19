#ifndef UTILS_H
#define UTILS_H

typedef struct batch {
    int to;
    int then;
} batch;

double random(double from, double to);
int max_index(double values[], int count);
//Double equals
int deq(double left, double right, double margin);
//Default double equals
int def_deq(double left, double right);
batch create_batch(int current, int size, int max);

#endif //UTILS_H
