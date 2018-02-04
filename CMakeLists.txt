cmake_minimum_required(VERSION 3.9.1)
project(SANN)

#########################DEFAULT MODE#########################
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif(NOT CMAKE_BUILD_TYPE)

#########################DATA SET#########################
if(NOT DEFINED DATA_SET)
    set(DATA_SET "monk1")
    set(DATA_FILE "example/monk.cpp")
elseif(${DATA_SET} STREQUAL "monk1" OR ${DATA_SET} STREQUAL "monk2" OR ${DATA_SET} STREQUAL "monk3")
    set(DATA_FILE "example/monk.cpp")
elseif(${DATA_SET} STREQUAL "cup")
    set(DATA_FILE "example/cup.cpp")
endif(NOT DEFINED DATA_SET)

#########################DIRECTORIES#########################
set(MYBASE_DIR src/sann)
set(MATH_DIR ${MYBASE_DIR}/math)
set(UTILITY_DIR ${MYBASE_DIR}/utility) 

#########################SOURCE FILES#########################
set(SANN_FILES ${MYBASE_DIR}/Network.cpp 
               ${MYBASE_DIR}/Layer.cpp
               ${MYBASE_DIR}/Network.cpp
               ${MYBASE_DIR}/Regularizer.cpp
               ${MYBASE_DIR}/Validator.cpp)
set(MATH_FILES ${MATH_DIR}/Func.cpp
               ${MATH_DIR}/Plotter.cpp
               ${MATH_DIR}/Randomizer.cpp)
set(UTILITY_FILES ${UTILITY_DIR}/FileManager.cpp
                  ${UTILITY_DIR}/Logger.cpp
                  ${UTILITY_DIR}/Stopwatch.cpp)
set(MAIN_FILE ${DATA_FILE})

#########################EXECUTABLE#########################
add_executable(${DATA_SET} ${MAIN_FILE} ${SANN_FILES} ${MATH_FILES} ${UTILITY_FILES})

#########################DEFINITION#########################
#add_definitions(-DS_DEBUG_MODE_S)

#########################CONFIGURE FILES#########################
set(CONFIGURE_DIR src/configure_files)
configure_file(${CONFIGURE_DIR}/constants.h.in /${CMAKE_CURRENT_SOURCE_DIR}/${MYBASE_DIR}/constants.h)

#########################COMPILER OPTIONS#########################
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set (CMAKE_CXX_FLAGS_DEBUG "-g")
set (CMAKE_CXX_FLAGS_RELEASE "-O3")

#########################BOOST#########################
find_package(Boost 1.62 COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
target_link_libraries(${DATA_SET} ${Boost_LIBRARIES})

#########################OPENMP#########################
find_package(OpenMP)
if (OPENMP_CXX_FOUND)
    set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()