cmake_minimum_required(VERSION 3.23)
project(NeuralNetwork C)
project(Sudoku)

set(CMAKE_C_STANDARD 23)

add_executable(NeuralNetwork neural_network.c functions.c repository.c generator.c utils.c main.c)
add_executable(Sudoku main.cpp)