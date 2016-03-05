CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(differential_evolution C CXX)

FIND_PACKAGE(Threads REQUIRED)

SET(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -std=c++11")
SET(CMAKE_CXX_FLAGS_DEBUG "-g")
SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_LINK_EXECUTABLE "-lboost_serialization -lboost_program_options")

INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR} ../../../src/binary_classifier/")
ADD_EXECUTABLE(pa passive_aggressive.cpp)
TARGET_LINK_LIBRARIES(pa ${CMAKE_LINK_EXECUTABLE})
