cmake_minimum_required(VERSION 3.23)
project(Test C)
project(Export C)

set(CMAKE_C_STANDARD 23)

add_executable(Test neural_network.c functions.c repository.c generator.c utils.c tests.c)
add_library(Export SHARED export.c neural_network.c functions.c utils.c repository.c)
