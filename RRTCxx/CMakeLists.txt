cmake_minimum_required(VERSION 3.15)
project(RRT LANGUAGES CXX)

# Pybind11 + Eigen
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

# COAL include + library (from conda or local)
set(COAL_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include")
set(COAL_LIBRARY "$ENV{CONDA_PREFIX}/lib/libcoal.so")


# 🔑 Tell CMake about COAL
include_directories(${COAL_INCLUDE_DIR})
link_directories("$ENV{CONDA_PREFIX}/lib")

# Build Pybind11 module
pybind11_add_module(_core src/rrtcxx/_core.cpp)
target_compile_options(_core PRIVATE -O3)


# Link libraries
target_link_libraries(_core PRIVATE Eigen3::Eigen ${COAL_LIBRARY})

# Install to your package structure
install(TARGETS _core DESTINATION rrtcxx)

# Force .so to appear inside src/rrtcxx/ for Python import
set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/rrtcxx
)
