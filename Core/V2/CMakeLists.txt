cmake_minimum_required(VERSION 3.30)
set(CMAKE_C_STANDARD 23)

option (FOR_DEVELOPMENT "For development" ON)

project(NewImplementation C)

set(SOURCES
        main.c
        cost.c
        logger.c
        builder.c
        neural_network.c
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
        Util/rand_util.c
        Optimizers/Types/simple_optimizer.c
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
        Optimizers/Types/free_momentum_optimizer.c
        Optimizers/Types/simplified_nesterov_optimizer.c
        Optimizers/Types/rms_prop_optimizer.c
        Optimizers/Types/adam_optimizer.c
        Optimizers/optimizer_factory.c
        DataSelector/data_selector_factory.c
        Schedulers/scheduler_factory.c
        Util/size.c
        Util/double_util.c
        Optimizers/Types/proportional_momentum_optimizer.c
        Optimizers/Types/adagrad_optimizer.c
        Optimizers/Types/adadelta_optimizer.c
        i_o.c
        Optimizers/Types/adamax_optimizer.c
        Optimizers/Types/adamw_optimizer.c
        Layers/Types/dropout_layer.c
        Layers/Types/dropout_layer.h)

if(MSVC)
    message(STATUS "MVSC detected - enabling CUDA")

    #To use CUDA in Windows, you need :
    #   - To use the MSVC compiler. Install Visual Studio 2022 Build Tools with the Desktop C++ kit.
    #   - To install the CUDA Toolkit and set the following variables
    set(CMAKE_CUDA_ARCHITECTURES 120)  #For RTX 5060
    set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0") #Edit with your path
    set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe")  #Edit with your path

    list(APPEND SOURCES Layers/Types/Cuda/cuda_dense_layer.cu)

    set(CMAKE_CXX_STANDARD 20)

    enable_language(CXX)
    enable_language(CUDA)

    if(FOR_DEVELOPMENT)
        message(STATUS "Setting up ASAN")

        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
    endif()
endif()

add_executable(NewImplementation ${SOURCES})

if(MSVC)
    target_include_directories(NewImplementation PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(NewImplementation PRIVATE ${CUDA_CUDART_LIBRARY})
endif()

add_custom_command(
        TARGET NewImplementation POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/Data
        $<TARGET_FILE_DIR:NewImplementation>/Data
)
