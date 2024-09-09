#ifndef REPOSITORY_H
#define REPOSITORY_H

struct neural_network* initialize(const char* file);
void save(const struct neural_network* network, const struct params* params, const char* file);

#endif // REPOSITORY_H
