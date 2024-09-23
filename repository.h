#ifndef REPOSITORY_H
#define REPOSITORY_H

neural_network* initialize(const char* file, params* toFill);
void save(const neural_network* network, const params* params, const char* file);

#endif // REPOSITORY_H
