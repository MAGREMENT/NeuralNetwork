cmake_minimum_required(VERSION 3.23)
project(Test C)

set(CMAKE_C_STANDARD 23)

add_executable(Test
        neural_network.c
        functions.c
        repository.c
        generator.c
        utils.c
        tests.c
        Iterators/iterator.h
        Iterators/Implementations/full_batch_iterator.c
        Iterators/Implementations/full_batch_iterator.h
        Iterators/Implementations/mini-batch_iterator.c
        Iterators/Implementations/mini-batch_iterator.h
        Iterators/iterator-util.c
        Iterators/iterator-util.h
        learning_rate_sechduler.c
        learning_rate_sechduler.h
        data_selector.c
        data_selector.h
        optimizer.c
        optimizer.h
        layer.c
        layer.h
        hyper_parameters.c
        hyper_parameters.h
        multi-threading.c
        multi-threading.h
        list.c
        list.h
        yaml.c
        yaml.h
        store.c)
add_library(Export SHARED
        export.c
        neural_network.c
        functions.c
        utils.c
        repository.c
        Iterators/iterator.h
        Iterators/Implementations/full_batch_iterator.c
        Iterators/Implementations/full_batch_iterator.h
        Iterators/Implementations/mini-batch_iterator.c
        Iterators/Implementations/mini-batch_iterator.h
        Iterators/iterator-util.c
        Iterators/iterator-util.h
        learning_rate_sechduler.c
        learning_rate_sechduler.h
        data_selector.c
        data_selector.h
        optimizer.c
        optimizer.h
        layer.c
        layer.h
        hyper_parameters.c
        hyper_parameters.h
        multi-threading.c
        multi-threading.h
        list.c
        list.h
        yaml.c
        yaml.h
        store.c)
