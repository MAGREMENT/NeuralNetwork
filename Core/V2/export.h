//
// Created by zacha on 29-11-25.
//

#ifndef NEWIMPLEMENTATION_EXPORT_H
#define NEWIMPLEMENTATION_EXPORT_H
#include "builder.h"
#include "neural_network.h"

#ifdef _WIN32
  #define DLL_EXPORT __declspec(dllexport)
#else
  #define DLL_EXPORT
#endif

DLL_EXPORT builder* CreateBuilder();
DLL_EXPORT void LoadBuilderFile(builder* builder, char file[]);
DLL_EXPORT neural_network* Build(builder* builder);
DLL_EXPORT void DisposeBuilder(builder* builder);
DLL_EXPORT void Dispose(neural_network* ptr);
DLL_EXPORT int GetLength(neural_network* ptr);
DLL_EXPORT int GetInCount(neural_network* ptr);
DLL_EXPORT int GetOutCount(neural_network* ptr);
DLL_EXPORT void LoadParameterFile(neural_network* ptr, char file[]);
DLL_EXPORT void Predict(neural_network* ptr, double* inputs, double* outputs);
DLL_EXPORT void Learn(neural_network* ptr, learning_data* state, double* inputs, double* expected, int count, int iterations);
DLL_EXPORT void LearnStateless(neural_network* ptr, double* inputs, double* expected, int count, int iterations);
DLL_EXPORT double Cost(neural_network* ptr, double* inputs, double* expected, int count);
DLL_EXPORT learning_data* CreateState(neural_network* ptr);
DLL_EXPORT void DisposeState(neural_network* ptr, learning_data* state);

#endif //NEWIMPLEMENTATION_EXPORT_H