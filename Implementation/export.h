#ifndef EXPORT_H
#define EXPORT_H

#include "neural_network.h"

neural_network* Initialize(int count, int numbers[]);
void Dispose(neural_network* ptr);
void ApplyParams(neural_network* ptr, params p);
int GetCount(const neural_network* ptr);
int GetOutCount(neural_network* ptr, int layer);
int GetInCount(neural_network* ptr, int layer);
void SetWeight(neural_network* ptr, int layer, int input, int output, double value);
double GetWeight(neural_network* ptr, int layer, int input, int output);
void SetBias(neural_network* ptr, int layer, int output, double value);
double GetBias(neural_network* ptr, int layer, int output);
void Predict(neural_network* ptr, double inputs[], int inCount, double outputs[], int outCount);
neural_network* FromFile(char file[], params* toFill);
void Save(neural_network* ptr, params p, char file[]);
void Randomize(neural_network* ptr, double min, double max);

#endif //EXPORT_H
