cmake_minimum_required(VERSION 3.23)
project(NeuralNetwork C)
project(Sudoku)
project(Mnist)

set(CMAKE_C_STANDARD 23)

add_executable(NeuralNetwork neural_network.c functions.c repository.c generator.c utils.c main.c)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Sudoku functions.c utils.c neural_network.c repository.c sudoku.cpp)
add_executable(Mnist functions.c utils.c neural_network.c repository.c mnist.cpp)

target_link_libraries(Sudoku ${OpenCV_LIBS})
target_link_libraries(Mnist ${OpenCV_LIBS})