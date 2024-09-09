cmake_minimum_required(VERSION 3.23)
project(NeuralNetwork C)

set(CMAKE_C_STANDARD 23)

add_executable(NeuralNetwork main.c neural_network.c functions.c repository.c
        generator.c utils.c)
