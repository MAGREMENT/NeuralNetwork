cmake_minimum_required(VERSION 3.30)
project(NewImplementation C)

set(CMAKE_C_STANDARD 23)

add_executable(NewImplementation main.c
        cost.c
        logger.c
        neural_network.c
        utils.c
        asserter.c
        Util/string_util.c
        Optimizers/optimizer.c
        Layers/layer.c
        Layers/Types/activation_layer.c
        Layers/Types/dense_layer.c
        Util/math_util.c
        Iterators/
        Iterators/iterator.c
        Iterators/Types/full_batch_iterator.c
        Iterators/Types/mini-batch_iterator.c
        multi-threading.c
        Layers/Types/convolutional_layer.c
        Util/Collections/list.c
        builder.c
        builder.h
        Util/rand_util.c
        Util/rand_util.h
        Optimizers/gradient_descent_optimizer.c
        Optimizers/gradient_descent_optimizer.h
)
