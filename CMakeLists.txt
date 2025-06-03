cmake_minimum_required(VERSION 3.10)
project(NeuralNetwork)

set(CMAKE_C_STANDARD 99)

add_executable(NeuralNetwork main.c
        mnist_gui.c)

# Добавляем линковку с математической библиотекой
target_link_libraries(NeuralNetwork m)
