#ifndef REPOSITORY_H
#define REPOSITORY_H

neural_network* restore(const char* file);
void save(const neural_network* network, const char* file);
void flog(char format[], ...);

#endif // REPOSITORY_H
