function(checked)
  execute_process(COMMAND ${ARGV} RESULT_VARIABLE result OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Consumer command failed (${result}): ${ARGV}\n${out}\n${err}")
  endif()
endfunction()
set(prefix "${VELOCITYPP_BUILD_DIR}/installed-test/prefix")
set(consumer "${VELOCITYPP_BUILD_DIR}/installed-test/consumer")
checked("${CMAKE_COMMAND}" --install "${VELOCITYPP_BUILD_DIR}" --prefix "${prefix}")
checked("${CMAKE_COMMAND}" -S "${VELOCITYPP_SOURCE_DIR}/tests/consumer" -B "${consumer}" -G Ninja
  "-DCMAKE_CXX_COMPILER=${VELOCITYPP_COMPILER}" "-DCMAKE_BUILD_TYPE=${VELOCITYPP_CONFIG}"
  "-DCMAKE_PREFIX_PATH=${prefix}\;${VELOCITYPP_ANNDATAPP_PREFIX}"
  "-DCONSUMER_SANITIZERS=${VELOCITYPP_SANITIZERS}")
checked("${CMAKE_COMMAND}" --build "${consumer}" --parallel 2)
checked("${consumer}/consumer")
checked("${consumer}/consumer_again")
