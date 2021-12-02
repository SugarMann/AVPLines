cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(BUILD_SYSTEM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../build_system")
include("${BUILD_SYSTEM_DIR}/cmake/init.cmake")


#-----------------------------------------------------------------------------------------
# Library name and version (Defined by user)
#   Check https://semver.org/ for library version numbering rules
#-----------------------------------------------------------------------------------------
project(AVPLines VERSION 0.1.0
                 DESCRIPTION "Detección de espazos de parking mediante visión por computador. Bachelor's Degree in Telecommunications Engineering Final Project - University of Vigo"
                 LANGUAGES CXX)


#-----------------------------------------------------------------------------------------
# Library sources
#	provides: ${PROJECT_NAME}_HEADER, ${PROJECT_NAME}_SRC, MOC_HEADERS,
#                 AUTOUIC_SEARCH_PATHS, QRC_SOURCES
#-----------------------------------------------------------------------------------------
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Public headers
set(${PROJECT_NAME}_HEADER
	${CMAKE_CURRENT_SOURCE_DIR}/include/vs/${PROJECT_NAME}/CFile.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vs/${PROJECT_NAME}/CMarkPoint.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vs/${PROJECT_NAME}/CSlot.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vs/${PROJECT_NAME}/main.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vs/${PROJECT_NAME}/json/json-forwards.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/vs/${PROJECT_NAME}/json/json.h
)

# Source files and non-public headers
set(${PROJECT_NAME}_SRC
	${CMAKE_CURRENT_SOURCE_DIR}/source/CFile.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/CMarkPoint.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/CSlot.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/jsoncpp.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/main.cpp
)


## Extra Qt stuff (MOC, UIC, RCC)

# Add .ui files
set(UI_SOURCES
        #${CMAKE_CURRENT_SOURCE_DIR}/resources/ui/prj_config_window.ui
)

# Add headers that need to be MOC'ed
# External headers that need to be MOC'ed need to be explicitly added.
set(MOC_HEADERS
#       ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/header1.hpp
#       ${BUILD_SYSTEM_DIR}/../releases/api/common/sensor_interface-0.1.0/include/vs/generic_sensor_interface.hpp
)

# Add resources (.qrc files)
set(QRC_SOURCES
#	${CMAKE_CURRENT_SOURCE_DIR}/resources/example1.qrc
)

# Needed ifthe .ui file is not in the same directory as the file that calls the generated header.
# NOTE: Only used if ActivateAutoMOC() is called.
set(AUTOUIC_SEARCH_PATHS
#        ${CMAKE_CURRENT_SOURCE_DIR}/resources/ui
)
#-----------------------------------------------------------------------------------------
# Libraries to be generated
#-----------------------------------------------------------------------------------------

GenerateLibrary()
GenerateExecutable()

#-----------------------------------------------------------------------------------------
# Create Config.cmake and ConfigVersion.cmake files
#-----------------------------------------------------------------------------------------
ConfigurePackage()

#-----------------------------------------------------------------------------------------
# Definitions of install targets libraries, headers and dev
#-----------------------------------------------------------------------------------------
InstallPackage()
	
#-----------------------------------------------------------------------------------------
# Test application
#-----------------------------------------------------------------------------------------
# include(CTest)
# if(BUILD_TESTING)
#     add_subdirectory(test)
# endif()

#-----------------------------------------------------------------------------------------
# Packaging
#-----------------------------------------------------------------------------------------
# ExportPackage()

