cmake_minimum_required(VERSION 3.15)
project(RRT LANGUAGES CXX)

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

set(COAL_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include")
set(COAL_LIBRARY     "$ENV{CONDA_PREFIX}/lib/libcoal.so")

pybind11_add_module(_core src/rrtcxx/_core.cpp)

target_link_libraries(_core PRIVATE Eigen3::Eigen ${COAL_LIBRARY})
install(TARGETS _core DESTINATION rrtcxx)

# ✅ Force output to match Python package layout
set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src/rrtcxx
)
