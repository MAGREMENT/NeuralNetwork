#ifndef REPOSITORY_H
#define REPOSITORY_H

neural_network* restore(const char* file);
int save(const neural_network* network, const char* file);

#endif // REPOSITORY_H
