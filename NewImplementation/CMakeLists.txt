cmake_minimum_required(VERSION 3.30)
project(NewImplementation C)

set(CMAKE_C_STANDARD 23)

add_executable(NewImplementation main.c
        conv_layer.c
        cost.c
        logger.c
        neural_network.c
        utils.c
        asserter.c
        Util/string_util.c)
