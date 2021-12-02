@echo off
setlocal

rem BATCH file that automatically runs a Vision Systems executable, compiled from a Build System-based project (`x64-Release` configuration) with a temporary/local PATH environment variable tailored to the "dependencies.txt" file of the project.
rem System/user Paths are NOT modified. No copy/pasting of DLLs is required.
rem The executable file to run is expected to be stored at the 'build_x64-Release/bin/Release/' directory, that is, where it is automatically generated when compiling.
rem The executable filename (with no directory) must be provided as first input parameter when invoking this script, followed by all the parameters required by the executable itself (e.g. `launch_project_exe vs_project_test.exe arg1 arg2`)
rem This script is supposed to be stored in the 'scripts/' directory at the root of the project repo.
rem Git for Windows (Git Bash) must be installed


rem  BUILD DIRECTORY LOCATION

set BUILD_DIR_RELEASE=..\build_x64-Release


rem INPUT PARAMETERS

set NUM_ARGS=0
for %%x in (%*) do set /A NUM_ARGS+=1
if %NUM_ARGS% == 0 (
    echo ERROR: The filename of the executable must be mandatorily provided as first argument
    goto end
) 

set EXE_FILE=%1
set EXE_COMMAND=%*


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


rem FIND BUILD DIRECTORY STRUCTURE

if not exist %BUILD_DIR_RELEASE%\ (
    echo ERROR: Directory %BUILD_DIR_RELEASE% does not exist. Project seems to be not generated nor compiled.
    echo        Please run `launch_bs_project.bat` and compile the `x64-Release` configuration before running this script again.
    goto end
) 

set BUILD_DIR_RELEASE_OUTPUT=%BUILD_DIR_RELEASE%\bin\Release\
if not exist %BUILD_DIR_RELEASE_OUTPUT%\ (
    echo ERROR: Directory %BUILD_DIR_RELEASE_OUTPUT% does not exist. Release configuration of the project seems to be not compiled.
    echo        Please run `launch_bs_project.bat` and compile the `x64-Release` configuration before running this script again.
    goto end
) 


rem FIND FILE TO EXECUTE

if not exist %BUILD_DIR_RELEASE_OUTPUT%\%EXE_FILE% (
    echo ERROR: There is no %EXE_FILE% at directory %BUILD_DIR_RELEASE_OUTPUT%
    goto end
)

echo FOUND FILE TO EXECUTE: %EXE_FILE% AT %BUILD_DIR_RELEASE_OUTPUT%


rem FIND THE SOLVED DEPENDENCIES FILE

set SOLVED_DEPENDENCIES_FILE=%BUILD_DIR_RELEASE%\solved_dependencies.txt

if not exist %SOLVED_DEPENDENCIES_FILE% (
    echo ERROR: File %SOLVED_DEPENDENCIES_FILE% does not exist.
    goto end
)
echo CAPTURING DEPENDENCIES FROM: %SOLVED_DEPENDENCIES_FILE%


rem PROCESS THE SOLVED DEPENDENCIES FILE

echo PROCESSING SOLVED DEPENDENCIES...
set DEPENDENCIES_STR=
for /F "tokens=1-3" %%i in ('type %SOLVED_DEPENDENCIES_FILE%') do call :process_dependencies %%i %%j %%k
goto run_command

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


rem RUN THE COMMAND

:run_command
echo RUNNING COMMAND "%BUILD_DIR_RELEASE_OUTPUT%\%EXE_COMMAND%" ...
%BUILD_DIR_RELEASE_OUTPUT%\%EXE_COMMAND%
goto end

 
:end
if exist new_dependencies_paths.tmp del new_dependencies_paths.tmp
endlocal