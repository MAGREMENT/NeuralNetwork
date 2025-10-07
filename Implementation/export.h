#ifndef EXPORT_H
#define EXPORT_H

#include "neural_network.h"

#ifdef _WIN32
  #define DLL_EXPORT __declspec(dllexport)
#else
  #define DLL_EXPORT
#endif

DLL_EXPORT neural_network* Create(int count, int numbers[]);
DLL_EXPORT void Dispose(neural_network* ptr);
DLL_EXPORT int GetCount(const neural_network* ptr);
DLL_EXPORT int GetOutCount(neural_network* ptr, int layer);
DLL_EXPORT int GetInCount(neural_network* ptr, int layer);
DLL_EXPORT void SetWeight(neural_network* ptr, int layer, int input, int output, double value);
DLL_EXPORT double GetWeight(neural_network* ptr, int layer, int input, int output);
DLL_EXPORT void SetBias(neural_network* ptr, int layer, int output, double value);
DLL_EXPORT double GetBias(neural_network* ptr, int layer, int output);
DLL_EXPORT void SetAllWeightsAndBiases(neural_network* ptr, double weights, double biases);
DLL_EXPORT double GetLearningRate(neural_network* n);
DLL_EXPORT void SetLearningRate(neural_network* n, double lr);
DLL_EXPORT int GetShuffleDataOnIteration(neural_network* n);
DLL_EXPORT void SetShuffleDataOnIteration(neural_network* n, int sdoi);
DLL_EXPORT void SetOptimizerGradientDescent(neural_network* n);
DLL_EXPORT void SetOptimizerMomentum(neural_network* n, double momentum);
DLL_EXPORT void SetOptimizerNesterov(neural_network* n, double decay);
DLL_EXPORT void SetOptimizerAdam(neural_network* n, double delta1, double delta2);
DLL_EXPORT void SetDataSelectorFullBatch(neural_network* n);
DLL_EXPORT void SetDataSelectorMiniBatch(neural_network* n, int batchSize);
DLL_EXPORT void Predict(neural_network* ptr, double inputs[], int inCount, double outputs[], int outCount);
DLL_EXPORT neural_network* FromFile(char file[]);
DLL_EXPORT void Save(neural_network* ptr, char file[]);
DLL_EXPORT void Initialize(neural_network* ptr);
DLL_EXPORT void Learn(neural_network* ptr, learning_state* state, double* inputs, int inputCutoff,
        double* expected, int expectedCutoff, int count, int iterations);
DLL_EXPORT double Cost(neural_network* ptr, double* inputs, int inputCount, double* expected, int expectedCount);
DLL_EXPORT double MultiCost(neural_network* ptr, double* inputs, int inputCutoff, double* expected, int expectedCutoff, int count);
DLL_EXPORT learning_state* CreateState(neural_network* ptr);
DLL_EXPORT void DisposeState(neural_network* ptr, learning_state* state);

#endif //EXPORT_H
