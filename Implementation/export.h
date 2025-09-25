#ifndef EXPORT_H
#define EXPORT_H

#include "neural_network.h"

#ifdef _WIN32
  #define DLL_EXPORT __declspec(dllexport)
#else
  #define DLL_EXPORT
#endif

DLL_EXPORT neural_network* Initialize(int count, int numbers[]);
DLL_EXPORT void Dispose(neural_network* ptr);
DLL_EXPORT void ApplyParams(neural_network* ptr, params p);
DLL_EXPORT int GetCount(const neural_network* ptr);
DLL_EXPORT int GetOutCount(neural_network* ptr, int layer);
DLL_EXPORT int GetInCount(neural_network* ptr, int layer);
DLL_EXPORT void SetWeight(neural_network* ptr, int layer, int input, int output, double value);
DLL_EXPORT double GetWeight(neural_network* ptr, int layer, int input, int output);
DLL_EXPORT void SetBias(neural_network* ptr, int layer, int output, double value);
DLL_EXPORT double GetBias(neural_network* ptr, int layer, int output);
DLL_EXPORT void Predict(neural_network* ptr, double inputs[], int inCount, double outputs[], int outCount);
DLL_EXPORT neural_network* FromFile(char file[], params* toFill);
DLL_EXPORT void Save(neural_network* ptr, params p, char file[]);
DLL_EXPORT void Randomize(neural_network* ptr, double min, double max);
DLL_EXPORT void Learn(neural_network* ptr, learning_state* state, double* inputs, int inputCutoff,
        double* expected, int expectedCutoff, int count, int batchSize, int iterations);
DLL_EXPORT double Cost(neural_network* ptr, double* inputs, int inputCount, double* expected, int expectedCount);
DLL_EXPORT double MultiCost(neural_network* ptr, double* inputs, int inputCutoff, double* expected, int expectedCutoff, int count);
DLL_EXPORT learning_state* InitializeState(neural_network* ptr, int batchSize);
DLL_EXPORT void DisposeState(learning_state* ptr, int layerCount);

#endif //EXPORT_H
