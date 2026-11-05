cmake_minimum_required(VERSION 3.28)
project(velocity VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18)
  message(FATAL_ERROR "Use Clang 18 or newer with matching clang-scan-deps.")
endif()
if(NOT CMAKE_GENERATOR MATCHES "Ninja")
  message(FATAL_ERROR "Use Ninja for C++20 module dependency scanning.")
endif()
if(NOT CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS)
  string(REGEX MATCH "^[0-9]+" _clang_major "${CMAKE_CXX_COMPILER_VERSION}")
  find_program(CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS NAMES clang-scan-deps-${_clang_major} clang-scan-deps REQUIRED)
endif()
find_package(anndatapp 0.1 CONFIG REQUIRED)
if(NOT TARGET Cellerator::binary16)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../ops/numeric/binary16/CMakeLists.txt")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../ops/numeric/binary16"
      "${CMAKE_CURRENT_BINARY_DIR}/binary16")
  else()
    find_package(CelleratorBinary16 CONFIG REQUIRED)
  endif()
endif()

option(VELOCITY_BUILD_TESTS "Build native and installed-consumer tests" ${PROJECT_IS_TOP_LEVEL})
option(VELOCITY_BUILD_EXAMPLES "Build portable CPU examples" ${PROJECT_IS_TOP_LEVEL})
option(VELOCITY_BUILD_QUALIFICATION "Build reference/benchmark drivers" OFF)
option(VELOCITY_ENABLE_SANITIZERS "Enable address and undefined-behavior sanitizers" OFF)
add_library(velocity)
add_library(velocity::velocity ALIAS velocity)
add_library(Cellerator::velocity ALIAS velocity)
set_target_properties(velocity PROPERTIES POSITION_INDEPENDENT_CODE ON EXPORT_NAME velocity_binary)
target_compile_features(velocity PUBLIC cxx_std_20)
target_sources(velocity PUBLIC FILE_SET CXX_MODULES BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src"
  FILES src/velocity.cppm src/velocity.numeric.cppm src/velocity.precision.cppm
  PRIVATE src/velocity.cpp src/numeric.cpp src/detail/prepared.cpp src/detail/fit.cpp src/detail/neighborhood.cpp
    src/detail/stochastic_fit.cpp src/detail/cpu_kernels.cpp src/precision.cpp)
target_link_libraries(velocity PUBLIC anndatapp::anndatapp Cellerator::binary16)
target_compile_options(velocity PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
if(VELOCITY_ENABLE_SANITIZERS)
  target_compile_options(velocity PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(velocity PUBLIC -fsanitize=address,undefined)
endif()

install(TARGETS velocity EXPORT velocityTargets
  FILE_SET CXX_MODULES DESTINATION ${CMAKE_INSTALL_DATADIR}/velocity/modules
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(EXPORT velocityTargets NAMESPACE velocity:: CXX_MODULES_DIRECTORY cxx-modules
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/velocity)
configure_package_config_file(cmake/velocityConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/velocityConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/velocity)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/velocityConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/velocityConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/velocityConfigVersion.cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/velocityModuleProvider.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/velocity)
install(FILES THIRD_PARTY_NOTICES.md DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/velocity)

if(VELOCITY_BUILD_EXAMPLES)
  add_executable(velocity_example examples/steady_state.cpp)
  target_link_libraries(velocity_example PRIVATE velocity::velocity)
  add_executable(velocity_stochastic_example examples/stochastic.cpp)
  target_link_libraries(velocity_stochastic_example PRIVATE velocity::velocity)
  add_executable(velocity_selective_example examples/selective_precision.cpp)
  target_link_libraries(velocity_selective_example PRIVATE velocity::velocity)
endif()
if(VELOCITY_BUILD_TESTS)
  enable_testing()
  foreach(suite reference customization stochastic stochastic_customization)
    add_executable(test_${suite} tests/test_${suite}.cpp)
    target_link_libraries(test_${suite} PRIVATE velocity::velocity)
    add_test(NAME ${suite} COMMAND test_${suite})
  endforeach()
  foreach(suite IN ITEMS cpu_kernels precision precision_cpu numeric_views)
    add_executable(test_${suite} tests/test_${suite}.cpp)
    target_link_libraries(test_${suite} PRIVATE velocity::velocity)
    add_test(NAME ${suite} COMMAND test_${suite})
  endforeach()
  if(VELOCITY_BUILD_EXAMPLES)
    add_test(NAME example COMMAND velocity_example)
    add_test(NAME stochastic-example COMMAND velocity_stochastic_example)
    add_test(NAME selective-example COMMAND velocity_selective_example)
  endif()
  add_test(NAME installed-consumer COMMAND ${CMAKE_COMMAND}
    -DVELOCITY_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
    -DVELOCITY_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}
    -DVELOCITY_COMPILER=${CMAKE_CXX_COMPILER}
    -DVELOCITY_CONFIG=${CMAKE_BUILD_TYPE}
    -DVELOCITY_ANNDATAPP_PREFIX=${anndatapp_DIR}/../../..
    -DVELOCITY_SANITIZERS=${VELOCITY_ENABLE_SANITIZERS}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test-installed-consumer.cmake)
endif()
if(VELOCITY_BUILD_QUALIFICATION)
  add_executable(velocity_precision_benchmark benchmarks/precision_qualification.cpp)
  target_link_libraries(velocity_precision_benchmark PRIVATE velocity::velocity)
  add_executable(velocity_sensitivity_driver tests/sensitivity_driver.cpp)
  target_link_libraries(velocity_sensitivity_driver PRIVATE velocity::velocity)
  add_executable(velocity_precision_interop_driver tests/precision_interop_driver.cpp)
  target_link_libraries(velocity_precision_interop_driver PRIVATE velocity::velocity)
  add_executable(velocity_interop_driver tests/interop_driver.cpp)
  target_link_libraries(velocity_interop_driver PRIVATE velocity::velocity)
  add_executable(velocity_benchmark benchmarks/qualification.cpp)
  target_link_libraries(velocity_benchmark PRIVATE velocity::velocity)
  add_executable(velocity_stochastic_interop_driver tests/stochastic_interop_driver.cpp)
  target_link_libraries(velocity_stochastic_interop_driver PRIVATE velocity::velocity)
  add_executable(velocity_stochastic_benchmark benchmarks/stochastic_qualification.cpp)
  target_link_libraries(velocity_stochastic_benchmark PRIVATE velocity::velocity)
endif()
