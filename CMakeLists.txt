cmake_minimum_required(VERSION 3.28)
project(velocitypp VERSION 0.1.0 LANGUAGES CXX)
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
if(NOT TARGET anndatapp::anndatapp)
  find_package(anndatapp 1.0 CONFIG REQUIRED)
endif()
option(VELOCITYPP_BUILD_TESTS "Build native and installed-consumer tests" ${PROJECT_IS_TOP_LEVEL})
option(VELOCITYPP_BUILD_EXAMPLES "Build portable CPU examples" ${PROJECT_IS_TOP_LEVEL})
option(VELOCITYPP_BUILD_QUALIFICATION "Build reference/benchmark drivers" OFF)
option(VELOCITYPP_ENABLE_SANITIZERS "Enable address and undefined-behavior sanitizers" OFF)
add_library(velocitypp)
add_library(velocitypp::velocitypp ALIAS velocitypp)
set_target_properties(velocitypp PROPERTIES POSITION_INDEPENDENT_CODE ON EXPORT_NAME velocitypp_binary)
target_compile_features(velocitypp PUBLIC cxx_std_20)
target_sources(velocitypp PUBLIC FILE_SET CXX_MODULES BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src"
  FILES src/velocitypp.cppm src/velocitypp.precision.cppm
  PRIVATE src/velocitypp.cpp src/detail/prepared.cpp src/detail/fit.cpp src/detail/neighborhood.cpp
    src/detail/stochastic_fit.cpp src/detail/cpu_kernels.cpp src/precision.cpp)
target_link_libraries(velocitypp PUBLIC
  $<BUILD_INTERFACE:anndatapp::anndatapp>
  $<INSTALL_INTERFACE:anndatapp::anndatapp>)
target_compile_options(velocitypp PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
if(VELOCITYPP_ENABLE_SANITIZERS)
  target_compile_options(velocitypp PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(velocitypp PUBLIC -fsanitize=address,undefined)
endif()

install(TARGETS velocitypp EXPORT velocityppTargets
  FILE_SET CXX_MODULES DESTINATION ${CMAKE_INSTALL_DATADIR}/velocitypp/modules
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(EXPORT velocityppTargets NAMESPACE velocitypp:: CXX_MODULES_DIRECTORY cxx-modules
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/velocitypp)
configure_package_config_file(cmake/velocityppConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/velocityppConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/velocitypp)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/velocityppConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/velocityppConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/velocityppConfigVersion.cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/velocityppModuleProvider.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/velocitypp)
install(FILES THIRD_PARTY_NOTICES.md DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/velocitypp)

if(VELOCITYPP_BUILD_EXAMPLES)
  add_executable(velocitypp_example examples/steady_state.cpp)
  target_link_libraries(velocitypp_example PRIVATE velocitypp::velocitypp)
  add_executable(velocitypp_stochastic_example examples/stochastic.cpp)
  target_link_libraries(velocitypp_stochastic_example PRIVATE velocitypp::velocitypp)
  add_executable(velocitypp_selective_example examples/selective_precision.cpp)
  target_link_libraries(velocitypp_selective_example PRIVATE velocitypp::velocitypp)
endif()
if(VELOCITYPP_BUILD_TESTS)
  enable_testing()
  foreach(suite reference customization stochastic stochastic_customization)
    add_executable(test_${suite} tests/test_${suite}.cpp)
    target_link_libraries(test_${suite} PRIVATE velocitypp::velocitypp)
    add_test(NAME ${suite} COMMAND test_${suite})
  endforeach()
  foreach(suite IN ITEMS cpu_kernels precision precision_cpu numeric_views)
    add_executable(test_${suite} tests/test_${suite}.cpp)
    target_link_libraries(test_${suite} PRIVATE velocitypp::velocitypp)
    add_test(NAME ${suite} COMMAND test_${suite})
  endforeach()
  if(VELOCITYPP_BUILD_EXAMPLES)
    add_test(NAME example COMMAND velocitypp_example)
    add_test(NAME stochastic-example COMMAND velocitypp_stochastic_example)
    add_test(NAME selective-example COMMAND velocitypp_selective_example)
  endif()
  add_test(NAME installed-consumer COMMAND ${CMAKE_COMMAND}
    -DVELOCITYPP_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
    -DVELOCITYPP_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}
    -DVELOCITYPP_COMPILER=${CMAKE_CXX_COMPILER}
    -DVELOCITYPP_CONFIG=${CMAKE_BUILD_TYPE}
    -DVELOCITYPP_ANNDATAPP_PREFIX=${anndatapp_DIR}/../../..
    -DVELOCITYPP_SANITIZERS=${VELOCITYPP_ENABLE_SANITIZERS}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/test-installed-consumer.cmake)
endif()
if(VELOCITYPP_BUILD_QUALIFICATION)
  add_executable(velocitypp_precision_benchmark benchmarks/precision_qualification.cpp)
  target_link_libraries(velocitypp_precision_benchmark PRIVATE velocitypp::velocitypp)
  add_executable(velocitypp_sensitivity_driver tests/sensitivity_driver.cpp)
  target_link_libraries(velocitypp_sensitivity_driver PRIVATE velocitypp::velocitypp)
  add_executable(velocitypp_precision_interop_driver tests/precision_interop_driver.cpp)
  target_link_libraries(velocitypp_precision_interop_driver PRIVATE velocitypp::velocitypp)
  add_executable(velocitypp_interop_driver tests/interop_driver.cpp)
  target_link_libraries(velocitypp_interop_driver PRIVATE velocitypp::velocitypp)
  add_executable(velocitypp_benchmark benchmarks/qualification.cpp)
  target_link_libraries(velocitypp_benchmark PRIVATE velocitypp::velocitypp)
  add_executable(velocitypp_stochastic_interop_driver tests/stochastic_interop_driver.cpp)
  target_link_libraries(velocitypp_stochastic_interop_driver PRIVATE velocitypp::velocitypp)
  add_executable(velocitypp_stochastic_benchmark benchmarks/stochastic_qualification.cpp)
  target_link_libraries(velocitypp_stochastic_benchmark PRIVATE velocitypp::velocitypp)
endif()
