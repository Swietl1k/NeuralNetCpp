cmake_minimum_required(VERSION 3.16)

project(NeuralNetCPP VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -pedantic -O2)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/W4)
endif()

# find Eigen 
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# source files
add_executable(neuralnet
    src/main.cpp
    src/mnist_loader.cpp
    src/network.cpp
)

# include directories
target_include_directories(neuralnet PRIVATE include)

# link Eigen
target_link_libraries(neuralnet PRIVATE Eigen3::Eigen)

# increase eigen stack allocation limit to 4MB (4194304 bytes)
target_compile_definitions(neuralnet PRIVATE 
    EIGEN_STACK_ALLOCATION_LIMIT=4194304
)

# set stack size to 16MB
if(APPLE)
    target_link_options(neuralnet PRIVATE 
        -Wl,-stack_size,0x1000000
    )
elseif(UNIX)
    target_link_options(neuralnet PRIVATE 
        -Wl,-z,stack-size=16777216
    )
endif()