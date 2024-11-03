cmake_minimum_required(VERSION 3.16...3.27)

project(mujoco_zenoh
    VERSION 0.0.1
    DESCRIPTION "MuJoCo Simulate with zenoh interface"
    LANGUAGES CXX C
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

find_package(zenohc REQUIRED)
find_package(zenohcxx REQUIRED)
find_package(spdlog REQUIRED)

add_library(mujoco::mujoco SHARED IMPORTED GLOBAL)

# Set the imported location and include directories
set_target_properties(mujoco::mujoco PROPERTIES
    IMPORTED_LOCATION "${mujoco_SOURCE_DIR}/lib/libmujoco.so.${MUJOCO_VERSION}"
    INTERFACE_INCLUDE_DIRECTORIES "${mujoco_SOURCE_DIR}/include"
    IMPORTED_NO_SONAME TRUE
)

list(APPEND CMAKE_MODULE_PATH "${mujoco_SOURCE_DIR}/simulate/cmake")

# Used for simulate to find the dependencies
set(MUJOCO_SIMULATE_USE_SYSTEM_GLFW ON)
include(SimulateOptions)
include(SimulateDependencies)

# Build libsimulate/lodepng
add_subdirectory(src/simulate)

add_executable(simulate src/main.cc ${mujoco_SOURCE_DIR}/simulate/array_safety.h)
target_compile_options(simulate PUBLIC ${MUJOCO_SIMULATE_COMPILE_OPTIONS})
target_link_libraries(
  simulate
  libsimulate
  mujoco::mujoco
  glfw
  Threads::Threads
  lodepng
)
target_link_options(simulate PRIVATE ${MUJOCO_SIMULATE_LINK_OPTIONS})
target_include_directories(simulate PRIVATE ${mujoco_SOURCE_DIR}/simulate)

add_executable(pub src/example.cpp)
target_link_libraries(
  pub
  zenohcxx::zenohc
)
add_executable(sub src/sub.cpp)
target_link_libraries(
  sub
  zenohcxx::zenohc
  spdlog::spdlog
)

install(TARGETS simulate
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
