CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(differential_evolution C CXX)

FIND_PACKAGE(Threads REQUIRED)

SET(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -std=c++14")
SET(CMAKE_CXX_FLAGS_DEBUG "-g")
SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_LINK_EXECUTABLE "-lboost_serialization -lboost_program_options")

ADD_EXECUTABLE(mpa mpa.cpp)
TARGET_LINK_LIBRARIES(mpa ${CMAKE_LINK_EXECUTABLE})
