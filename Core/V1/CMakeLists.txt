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
        Iterators/Implementations/full_batch_iterator.c
        Iterators/Implementations/mini-batch_iterator.c
        Iterators/iterator-util.c
        learning_rate_sechduler.c
        data_selector.c
        optimizer.c
        layer.c
        hyper_parameters.c
        multi-threading.c
        list.c
        yaml.c
        store.c
        logger.c
)
add_library(Export SHARED
        export.c
        neural_network.c
        functions.c
        utils.c
        repository.c
        Iterators/Implementations/full_batch_iterator.c
        Iterators/Implementations/mini-batch_iterator.c
        Iterators/iterator-util.c
        learning_rate_sechduler.c
        data_selector.c
        optimizer.c
        layer.c
        hyper_parameters.c
        multi-threading.c
        list.c
        yaml.c
        store.c
        logger.c
)
