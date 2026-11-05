function(checked)
  execute_process(COMMAND ${ARGV} RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Consumer command failed (${result}): ${ARGV}\n${out}\n${err}")
  endif()
endfunction()
set(prefix "${VELOCITY_BUILD_DIR}/installed-test/prefix")
set(consumer "${VELOCITY_BUILD_DIR}/installed-test/consumer")
checked("${CMAKE_COMMAND}" --install "${VELOCITY_BUILD_DIR}" --prefix "${prefix}")
checked("${CMAKE_COMMAND}" -S "${VELOCITY_SOURCE_DIR}/tests/consumer" -B "${consumer}" -G Ninja
  "-DCMAKE_CXX_COMPILER=${VELOCITY_COMPILER}" "-DCMAKE_BUILD_TYPE=${VELOCITY_CONFIG}"
  "-DCMAKE_PREFIX_PATH=${prefix}\;${VELOCITY_ANNDATAPP_PREFIX}"
  "-DCONSUMER_SANITIZERS=${VELOCITY_SANITIZERS}")
checked("${CMAKE_COMMAND}" --build "${consumer}" --parallel 2)
checked("${consumer}/consumer")
checked("${consumer}/consumer_again")
