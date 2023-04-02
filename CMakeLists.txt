cmake_minimum_required(VERSION 3.10.0)
project(plearn VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
# add_compile_options(-Wall -Wextra -Wpedantic -Werror)
add_compile_options(-Wextra -Wpedantic)
add_compile_options(-Wunknown-pragmas -Wno-narrowing)
cmake_policy(SET CMP0069 NEW)
include(CheckIPOSupported)
check_ipo_supported()
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")


if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE EQUAL "Debug")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb3 -O0")
endif()


# CONAN
set(CONAN_DEP_PATH ${PROJECT_SOURCE_DIR}/)
execute_process(COMMAND conan install ${CONAN_DEP_PATH}
	WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
	)
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_library(plearn_msg STATIC
	include/data/gen/call_graph.pb.cc
	)
target_link_libraries(plearn_msg PUBLIC CONAN_PKG::protobuf)

add_library(plearn_core INTERFACE)
target_include_directories(plearn_core INTERFACE include)
target_link_libraries(plearn_core INTERFACE
	${CONAN_LIBS}
	plearn_msg
	)

if (PLEARN_TEST)
	conan_basic_setup(TARGETS)
	enable_testing()
	include(GoogleTest)
	add_executable(unit_test 
		test/cmpl/basic_test.cpp
		test/cmpl/layer_test.cpp

		test/rep/call_graph_test.cpp
		test/rep/forward_prop_test.cpp

		test/environ/exec_env_test.cpp
		test/environ/diff_env_test.cpp

		test/backend/cpu/cpu_ops_test.cpp
		test/backend/cpu/cpu_fp_chain_grad_test.cpp
		test/backend/cpu/cpu_integration_test.cpp

		test/data/persistence_test.cpp
		)
	target_link_libraries(unit_test PRIVATE CONAN_PKG::gtest plearn_core)
	gtest_discover_tests(unit_test)
endif()
