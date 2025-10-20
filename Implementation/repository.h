#ifndef REPOSITORY_H
#define REPOSITORY_H

old_nn* restore(const char* file);
int save(const old_nn* network, const char* file);
void flog(char format[], ...);

#endif // REPOSITORY_H
