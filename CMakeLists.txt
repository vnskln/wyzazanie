cmake_minimum_required(VERSION 3.20)
project(rozproszone)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find MPI package
find_package(MPI REQUIRED)

add_executable(rozproszone main.cpp algorithm.cpp functions.cpp)

# Link MPI libraries
target_link_libraries(rozproszone PRIVATE MPI::MPI_CXX)
target_include_directories(rozproszone PRIVATE ${MPI_CXX_INCLUDE_DIRS})
