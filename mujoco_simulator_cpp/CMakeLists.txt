cmake_minimum_required(VERSION 3.16...3.27)

project(mujoco_zenoh
    VERSION 0.0.1
    DESCRIPTION "MuJoCo Simulate with zenoh interface"
    LANGUAGES CXX C
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(mujoco REQUIRED)

include(FetchContent)
include(GNUInstallDirs)

set(MUJOCO_VERSION "3.2.4")

set(FETCHCONTENT_QUIET OFF)
# Download and extract MuJoCo
FetchContent_Declare(
    mujoco
    URL "https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz"
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(mujoco)

find_package(range-v3 REQUIRED)
find_package(zenohc REQUIRED)
find_package(zenohcxx REQUIRED)
find_package(spdlog REQUIRED)

# add_library(mujoco::mujoco SHARED IMPORTED GLOBAL)

# Set the imported location and include directories
# set_target_properties(mujoco::mujoco PROPERTIES
#     IMPORTED_LOCATION "${mujoco_SOURCE_DIR}/lib/libmujoco.so.${MUJOCO_VERSION}"
#     INTERFACE_INCLUDE_DIRECTORIES "${mujoco_SOURCE_DIR}/include"
#     IMPORTED_NO_SONAME TRUE
# )

list(APPEND CMAKE_MODULE_PATH "${mujoco_SOURCE_DIR}/simulate/cmake")

# Used for simulate to find the dependencies
set(MUJOCO_SIMULATE_USE_SYSTEM_GLFW ON)
set(MUJOCO_SIMULATE_USE_SYSTEM_MUJOCO ON)
include(SimulateOptions)
include(SimulateDependencies)

# Build libsimulate/lodepng
add_subdirectory(src/simulate)

add_executable(mujoco_simulator src/main.cc ${mujoco_SOURCE_DIR}/simulate/array_safety.h)
target_compile_features(mujoco_simulator PRIVATE cxx_std_20)
target_compile_options(mujoco_simulator PUBLIC ${MUJOCO_SIMULATE_COMPILE_OPTIONS})
target_link_libraries(
  mujoco_simulator
  libsimulate
  mujoco::mujoco
  glfw
  Threads::Threads
  lodepng
  spdlog::spdlog
  zenohcxx::zenohc
)
target_link_options(mujoco_simulator PRIVATE ${MUJOCO_SIMULATE_LINK_OPTIONS})
target_include_directories(mujoco_simulator PRIVATE ${mujoco_SOURCE_DIR}/simulate include)

# add_executable(mujoco_model_view src/mujoco_model_view.cpp)
# target_compile_features(mujoco_model_view PRIVATE cxx_std_20)
# target_include_directories(mujoco_model_view PRIVATE include)
# target_link_libraries(
#   mujoco_model_view
#   mujoco::mujoco
#   spdlog::spdlog
# )

install(TARGETS mujoco_simulator
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
