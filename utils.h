#ifndef UTILS_H
#define UTILS_H

struct batch {
    int to;
    int then;
};

double random(double from, double to);
int max_index(double values[], int count);
int equals(double left, double right, double margin);
int default_equals(double left, double right);
struct batch create_batch(int current, int size, int max);

#endif //UTILS_H
