
#-----------------------------------------------------------------------------------------
# Test sources
#	provides: ${PROJECT_NAME}_HEADER_TEST, ${PROJECT_NAME}_SRC_TEST, MOC_HEADERS_TEST,
#                 UI_SOURCES_TEST, QRC_SOURCES_TEST AUTOUIC_SEARCH_PATHS_TEST
#-----------------------------------------------------------------------------------------

# Public headers
set(${PROJECT_NAME}_HEADER_TEST

)

# Source files and non-public headers
set(${PROJECT_NAME}_SRC_TEST
	${CMAKE_CURRENT_SOURCE_DIR}/source/main.cpp
)


## Extra Qt stuff (MOC, UIC, RCC), can be deleted if unused

# Add .ui files
set(UI_SOURCES_TEST
        #${CMAKE_CURRENT_SOURCE_DIR}/resources/ui/prj_config_window.ui
)

# Add headers that need to be MOC'ed
# External headers that need to be MOC'ed need to be explicitly added.
set(MOC_HEADERS_TEST
#       ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/header1.hpp
#       ${BUILD_SYSTEM_DIR}/../releases/api/common/sensor_interface-0.1.0/include/vs/generic_sensor_interface.hpp
)

# Add resources (.qrc files)
set(QRC_SOURCES_TEST
#	${CMAKE_CURRENT_SOURCE_DIR}/resources/example1.qrc
)

# Needed ifthe .ui file is not in the same directory as the file that calls the generated header.
# NOTE: Only used if ActivateAutoMOC() is called.
set(AUTOUIC_SEARCH_PATHS_TEST
#        ${CMAKE_CURRENT_SOURCE_DIR}/resources/ui
)

#-----------------------------------------------------------------------------------------
# Targets to be generated
#-----------------------------------------------------------------------------------------
GenerateTest()



