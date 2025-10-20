#ifndef EXPORT_H
#define EXPORT_H

#include "old_nn.h"

#ifdef _WIN32
  #define DLL_EXPORT __declspec(dllexport)
#else
  #define DLL_EXPORT
#endif

DLL_EXPORT old_nn* Create(int count, int numbers[]);
DLL_EXPORT void Dispose(old_nn* ptr);
DLL_EXPORT int GetCount(const old_nn* ptr);
DLL_EXPORT int GetOutCount(old_nn* ptr, int layer);
DLL_EXPORT int GetInCount(old_nn* ptr, int layer);
DLL_EXPORT void SetWeight(old_nn* ptr, int layer, int input, int output, double value);
DLL_EXPORT double GetWeight(old_nn* ptr, int layer, int input, int output);
DLL_EXPORT void SetBias(old_nn* ptr, int layer, int output, double value);
DLL_EXPORT double GetBias(old_nn* ptr, int layer, int output);
DLL_EXPORT void SetAllWeightsAndBiases(old_nn* ptr, double weights, double biases);
DLL_EXPORT double GetLearningRate(old_nn* n);
DLL_EXPORT void SetLearningRate(old_nn* n, double lr);
DLL_EXPORT int GetShuffleDataOnIteration(old_nn* n);
DLL_EXPORT void SetShuffleDataOnIteration(old_nn* n, int sdoi);
DLL_EXPORT int GetThreadCount(old_nn* n);
DLL_EXPORT void SetThreadCount(old_nn* n, int th);
DLL_EXPORT void SetActivationType(old_nn* n, int type, int outputType);
DLL_EXPORT void SetCostType(old_nn* n, int type);
DLL_EXPORT void SetOptimizerGradientDescent(old_nn* n);
DLL_EXPORT void SetOptimizerMomentum(old_nn* n, double momentum);
DLL_EXPORT void SetOptimizerNesterov(old_nn* n, double decay);
DLL_EXPORT void SetOptimizerAdam(old_nn* n, double delta1, double delta2);
DLL_EXPORT void SetDataSelectorFullBatch(old_nn* n);
DLL_EXPORT void SetDataSelectorMiniBatch(old_nn* n, int batchSize);
DLL_EXPORT void SetSchedulerConstant(old_nn* n);
DLL_EXPORT void SetSchedulerIterationDecay(old_nn* n, double proportion);
DLL_EXPORT void SetSchedulerExponentialDecay(old_nn* n, double decay);
DLL_EXPORT void SetSchedulerInverseDecay(old_nn* n, double decay);
DLL_EXPORT void SetSchedulerCosineDecay(old_nn* n, double endLearningRate, int iterationSpan);
DLL_EXPORT void Predict(old_nn* ptr, double inputs[], int inCount, double outputs[], int outCount);
DLL_EXPORT old_nn* FromFile(char file[]);
DLL_EXPORT int Save(old_nn* ptr, char file[]);
DLL_EXPORT void Initialize(old_nn* ptr);
DLL_EXPORT void Learn(old_nn* ptr, learning_state* state, double* inputs, int inputCutoff,
        double* expected, int expectedCutoff, int count, int iterations);
DLL_EXPORT double Cost(old_nn* ptr, double* inputs, int inputCount, double* expected, int expectedCount);
DLL_EXPORT double MultiCost(old_nn* ptr, double* inputs, int inputCutoff, double* expected, int expectedCutoff, int count);
DLL_EXPORT learning_state* CreateState(old_nn* ptr);
DLL_EXPORT void DisposeState(old_nn* ptr, learning_state* state);
DLL_EXPORT void Standardize(double* inputs, int inputCutoff, double* expected, int expectedCutoff, int count);
DLL_EXPORT void MinMaxScale(double* inputs, int inputCutoff, double* expected, int expectedCutoff, int count);

#endif //EXPORT_H
