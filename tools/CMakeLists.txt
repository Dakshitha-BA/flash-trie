cmake_minimum_required(VERSION 3.14)
project(marisa_tools LANGUAGES CXX)

option(FLASHTRIE_BUILD_TOOLS "Build command-line tools" ON)
option(FLASHTRIE_BUILD_PYTHON "Build the flashtrie Python extension" ON)

if(NOT FLASHTRIE_CUDA_INCLUDE)
  set(FLASHTRIE_CUDA_INCLUDE /usr/local/cuda/include)
endif()

if(FLASHTRIE_BUILD_TOOLS)
  set(PROGS
    marisa-build
    marisa-lookup
    marisa-reverse-lookup
    marisa-common-prefix-search
    marisa-predictive-search
    marisa-dump
    marisa-benchmark
    marisa-beam-search
  )

  foreach(prog ${PROGS})
    add_executable(${prog} ${prog}.cc cmdopt.cc)
    target_include_directories(${prog} PUBLIC ${FLASHTRIE_CUDA_INCLUDE})
    target_link_libraries(${prog} PRIVATE marisa)
  endforeach()
endif()

if(FLASHTRIE_BUILD_PYTHON)
  if(NOT pybind11_DIR)
    find_package(Python COMPONENTS Interpreter QUIET)
    if(Python_EXECUTABLE)
      execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _pybind11_cmake
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _pybind11_rc
      )
      if(_pybind11_rc EQUAL 0 AND _pybind11_cmake)
        set(pybind11_DIR "${_pybind11_cmake}")
      endif()
    endif()
  endif()
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(_core bindings.cpp)
  target_compile_definitions(_core PRIVATE "FLASHTRIE_VERSION=\"${FLASHTRIE_VERSION}\"")
  target_compile_options(_core PRIVATE -O3 -std=c++17 -Wall -Wextra -fmax-errors=1)
  target_include_directories(_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${FLASHTRIE_CUDA_INCLUDE}
  )
  target_link_libraries(_core PRIVATE marisa)

  # pip install sets this so the extension lands in the package directory.
  if(FLASHTRIE_PYTHON_OUTPUT_DIR)
    set_target_properties(_core PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${FLASHTRIE_PYTHON_OUTPUT_DIR}"
      LIBRARY_OUTPUT_DIRECTORY_RELEASE "${FLASHTRIE_PYTHON_OUTPUT_DIR}"
      LIBRARY_OUTPUT_DIRECTORY_DEBUG "${FLASHTRIE_PYTHON_OUTPUT_DIR}"
      RUNTIME_OUTPUT_DIRECTORY "${FLASHTRIE_PYTHON_OUTPUT_DIR}"
      RUNTIME_OUTPUT_DIRECTORY_RELEASE "${FLASHTRIE_PYTHON_OUTPUT_DIR}"
      RUNTIME_OUTPUT_DIRECTORY_DEBUG "${FLASHTRIE_PYTHON_OUTPUT_DIR}"
    )
  endif()
endif()
