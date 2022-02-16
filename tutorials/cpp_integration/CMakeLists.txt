cmake_minimum_required(VERSION 3.16.3)
project(cpp_integration)

# Python
find_package( PythonLibs   3 )
find_package( Python3 REQUIRED COMPONENTS Interpreter Development )
set( PYTHON_VERSION ${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR} )

# Boost
find_package(Boost COMPONENTS
  python${PYTHON_VERSION}
  numpy${PYTHON_VERSION}
)

if( NOT Boost_FOUND )
  ## Fallback to pythonMajor
  find_package( Boost REQUIRED COMPONENTS
    python${PYTHON_VERSION_MAJOR}
    numpy${PYTHON_VERSION_MAJOR}
  )
endif()

# OpenCV
find_package(OpenCV REQUIRED )

# CXX_STANDARD
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

###################
## Build library ##
###################
set(PROJECT_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(PROJECT_LIBRARIES ${PROJECT_NAME})

# Data
set(data_lib
  include/data/Hole.hpp
  src/data/Hole.cpp
)

# Detector
set(detector_lib
  include/detector/DnnHoleLocalizer.hpp
  src/detector/DnnHoleLocalizer.cpp
)

# Gather all libs
set(srcs_lib
  ${data_lib}
  ${detector_lib}
)

add_library(${PROJECT_LIBRARIES} ${srcs_lib})
target_link_libraries(${PROJECT_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${OpenCV_LIBS})
include_directories(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})

########################################
## Install python_deeplearning module ##
########################################
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/deeplearning/python_deeplearning /tmp/python/python_deeplearning)

###########################
## Build tutorials ##
###########################
set(srcs_tutorials
  tutorial_cpp_integration.cpp
)

foreach(cpp ${srcs_tutorials})
  get_filename_component(target ${cpp} NAME_WE)
  include_directories(${Boost_INCLUDE_DIRS} ${PROJECT_INCLUDE_DIRS})
  add_executable("${target}" ${cpp})
  target_link_libraries("${target}" ${PROJECT_LIBRARIES})
endforeach()
