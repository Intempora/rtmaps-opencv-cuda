##############################################################################
# 
#  Copyright 2014-2025 Intempora S.A.S.
#  
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#      http://www.apache.org/licenses/LICENSE-2.0
#  
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
##############################################################################
cmake_minimum_required(VERSION 3.0)

set(PCK rtmaps_image_processing_opencv_cuda)
project(${PCK})

if (NOT DEFINED RTMAPS_SDKDIR)  # If -D"RTMAPS_SDKDIR=<RTMaps Install Dir>" has NOT been passed to the "cmake" command
    if (DEFINED ENV{RTMAPS_SDKDIR})  # Try to use the default RTMaps installation
        set(RTMAPS_SDKDIR "$ENV{RTMAPS_SDKDIR}" CACHE PATH "Path the RTMaps installation directory")
    else()
        message(FATAL_ERROR "RTMAPS_SDKDIR not defined. Either pass -D\"RTMAPS_SDKDIR=...\" to CMake or define an RTMAPS_SDKDIR environment variable")
    endif()
endif()

include("${RTMAPS_SDKDIR}/templates.u/rtmaps.cmake")

if (NOT DEFINED OPENCV_PATH)
    message(FATAL_ERROR "You have to specify the path of your opencv installation with -DOPENCV_PATH=...")
endif()

if (NOT DEFINED USE_OPENCV_STATIC)
    set(OpenCV_STATIC OFF)
else()
    set(OpenCV_STATIC ${USE_OPENCV_STATIC})
endif()

find_package(OpenCV 4 REQUIRED PATHS ${OPENCV_PATH})  # Specify the DIRECTORY of OpenCVConfig.cmake

find_package(CUDA REQUIRED)

add_rtmaps_package(${PCK} PCKINFO "${PCK}.pckinfo"
    "local_interfaces"
    "local_interfaces/common"
    "src"
)

target_include_directories(${PCK} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(${PCK}
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    rtmaps_input_reader
)
