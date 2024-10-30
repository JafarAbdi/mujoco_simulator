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

set(MUJOCO_SIMULATE_USE_SYSTEM_GLFW ON)
include(SimulateDependencies)
include(SimulateOptions)

add_executable(basic ${mujoco_SOURCE_DIR}/sample/basic.cc)
target_compile_options(basic PUBLIC ${MUJOCO_SAMPLE_COMPILE_OPTIONS})
target_link_libraries(
  basic
  mujoco::mujoco
  glfw
  Threads::Threads
)
target_link_options(basic PRIVATE ${MUJOCO_SAMPLE_LINK_OPTIONS})

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

install(TARGETS pub sub basic
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# enable_testing()
# add_test(NAME ${PROJECT_NAME}_test 
#          COMMAND ${PROJECT_NAME} --version)
