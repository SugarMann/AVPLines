@echo off
setlocal

rem BATCH file that automatically configures and opens a Vision Systems Build-System-based project in the Visual Studio IDE with a temporary/local PATH environment variable tailored to the "dependencies.txt" file of the project.
rem It allows to debug/run our applications directly from the IDE, having the right denpendencies fully controlled.
rem System/user Paths are NOT modified. No copy/pasting of DLLs is required.
rem When necessary (for example, in a "just cloned" repo), it also automatically generates the CMake project according to the Build System.
rem This script is supposed to be stored in the 'scripts' directory at the root of the project repo.
rem Compatible with Visual Studio 2017 Community and Professional
rem Git for Windows (Git Bash) must be installed


rem IDE VARIABLES

set VS_2017_COMMUNITY_EXE="C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\IDE\devenv.exe"
set VS_2017_COMMUNITY_CMAKE_EXE="C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set VS_2017_COMMUNITY_CMAKE_GENERATOR="Visual Studio 15 2017 Win64"

set VS_2017_PROFESSIONAL_EXE="C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\Common7\IDE\devenv.exe"
set VS_2017_PROFESSIONAL_CMAKE_EXE="C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set VS_2017_PROFESSIONAL_CMAKE_GENERATOR="Visual Studio 15 2017 Win64"


rem FIND A COMPATIBLE IDE

set IDE_LAUNCHER="EMPTY"
if exist %VS_2017_PROFESSIONAL_EXE% (
    set IDE_LAUNCHER=%VS_2017_PROFESSIONAL_EXE%
    set CMAKE_EXE=%VS_2017_PROFESSIONAL_CMAKE_EXE%
    set CMAKE_GENERATOR=%VS_2017_PROFESSIONAL_CMAKE_GENERATOR%
)
if exist %VS_2017_COMMUNITY_EXE% (
    set IDE_LAUNCHER=%VS_2017_COMMUNITY_EXE%
    set CMAKE_EXE=%VS_2017_COMMUNITY_CMAKE_EXE%
    set CMAKE_GENERATOR=%VS_2017_COMMUNITY_CMAKE_GENERATOR%
)

if %IDE_LAUNCHER% == "EMPTY" (
    echo ERROR: No compatible IDE has been found
    goto end
) 
echo SETTING GOAL IDE TO: %IDE_LAUNCHER%


rem BUILD FOLDERS LOCATION

set BUILD_DIR_RELEASE=..\build_x64-Release


rem FIND GIT FOR WINDOWS INSTALLATION

set GIT_SH_EXE="EMPTY"
if exist "%USERPROFILE%\AppData\Local\Programs\Git\bin\sh.exe" (
    set GIT_SH_EXE="%USERPROFILE%\AppData\Local\Programs\Git\bin\sh.exe"
)
if exist "%PROGRAMFILES%\Git\bin\sh.exe" (
    set GIT_SH_EXE="%PROGRAMFILES%\Git\bin\sh.exe"
)
if %GIT_SH_EXE% == "EMPTY" (
    echo ERROR: No Git for Windows Installation has been found
    goto end
) 


rem FIND THE SOLVED DEPENDENCIES FILE

if not exist %BUILD_DIR_RELEASE%\ mkdir %BUILD_DIR_RELEASE%
set SOLVED_DEPENDENCIES_FILE=%BUILD_DIR_RELEASE%\solved_dependencies.txt
    
if not exist %SOLVED_DEPENDENCIES_FILE% (
    echo WARNING: File %SOLVED_DEPENDENCIES_FILE% does not exist, the CMake project will be [re]generated.
    pushd %BUILD_DIR_RELEASE%
    %CMAKE_EXE% -G %CMAKE_GENERATOR% -DCMAKE_INSTALL_PREFIX:PATH=..\install\ -D CONFIG=Release -D ARCH=win64 -D BUILD_SHARED_LIBS=ON -D CMAKE_BUILD_TYPE=Release -DCMAKE_CONFIGURATION_TYPES=Release ..
    popd
)

if not exist %SOLVED_DEPENDENCIES_FILE% (
    echo ERROR: File %SOLVED_DEPENDENCIES_FILE% could not be created
    goto end
)

echo CAPTURING DEPENDENCIES FROM: %SOLVED_DEPENDENCIES_FILE%


rem PROCESS THE SOLVED DEPENDENCIES FILE

echo PROCESSING SOLVED DEPENDENCIES...
set DEPENDENCIES_STR=
for /F "tokens=1-3" %%i in ('type %SOLVED_DEPENDENCIES_FILE%') do call :process_dependencies %%i %%j %%k
goto launch_ide

:process_dependencies
set LIBRARY_NAME=%1
set LIBRARY_VERSION=%2
set FOLDER_TO_SEARCH=%3
echo PROCESSING %LIBRARY_NAME% (%LIBRARY_VERSION%)...
if "%LIBRARY_NAME:~0,3%" == "vs_" (
    set FOLDER_TO_SEARCH=%FOLDER_TO_SEARCH%/..
)
if exist %FOLDER_TO_SEARCH%\ goto find_dll_subfolders
goto end

:find_dll_subfolders
echo SEARCHING DLL FOLDERS IN "%FOLDER_TO_SEARCH%"...
set COMMAND="find %FOLDER_TO_SEARCH% -xtype f -name '*.dll' | sed -r 's|/[^/]+$||'  | sort | uniq  | tr '/' '\\\\\\\\' > new_dependencies_paths.tmp"
%GIT_SH_EXE% -i -c %COMMAND%
echo UPDATING PATH...

for /F "tokens=*" %%i in ('type new_dependencies_paths.tmp') do call :update_path %%i
goto end

:update_path
set NEW_DIRECTORY=%1
set PATH=%NEW_DIRECTORY%;%PATH%
goto end


rem LAUNCH PROJECT IN IDE

:launch_ide
echo LAUNCHING PROJECT IN IDE...
%IDE_LAUNCHER% ../
goto end


:end
if exist new_dependencies_paths.tmp del new_dependencies_paths.tmp
endlocal