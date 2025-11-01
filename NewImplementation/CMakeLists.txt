cmake_minimum_required(VERSION 3.30)
project(NewImplementation C)

set(CMAKE_C_STANDARD 23)

add_executable(NewImplementation
        main.c
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
        Iterators/iterator.c
        Iterators/Types/full_batch_iterator.c
        Iterators/Types/mini-batch_iterator.c
        multi-threading.c
        Layers/Types/convolutional_layer.c
        Util/Collections/list.c
        builder.c
        Util/rand_util.c
        Optimizers/Types/gradient_descent_optimizer.c
        Schedulers/scheduler.c
        DataSelector/data_selector.c
        DataSelector/Types/full_batch_data_selector.c
        DataSelector/Types/mini_batch_data_selector.c
        Schedulers/Types/constant_scheduler.c
        Schedulers/Types/iteration_decay_scheduler.c
        Schedulers/Types/exponential_decay_scheduler.c
        Schedulers/Types/inverse_decay_scheduler.c
        Schedulers/Types/cosine_decay_scheduler.c
        Layers/Types/pooling_layer.c
        Optimizers/Types/momentum_gradient_descent_optimizer.c
        Optimizers/Types/nesterov_optimizer.c
        Optimizers/Types/rms_prop_optimizer.c
        Optimizers/Types/adam_optimizer.c
        Optimizers/optimizer_factory.c
        DataSelector/data_selector_factory.c
        Schedulers/scheduler_factory.c
)
