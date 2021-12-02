@echo off
setlocal

rem BATCH file that automatically runs a Vision Systems executable file, with a temporary/local PATH environment variable including all the Vision System dependencies (`third_party` and `releases`) available in the local system.
rem System/user Paths are NOT modified. No copy/pasting of DLLs is required.
rem The path of the executable file to run must be provided as first input parameter when invoking this script, followed by all the parameters required by the executable itself (e.g. `launch_exe vs_project_test.exe arg1 arg2`)
rem This script can be stored anywhere in the system
rem Git for Windows (Git Bash) must be installed


rem  FIND VISIONS SYSTEMS REPOS DIRECTORY LOCATION FROM BUILD_SYSTEM_DIR ENVIRONMENT VARIABLE

if "%BUILD_SYSTEM_DIR%"=="" (
    echo ERROR: `BUILD_SYSTEM_DIR` environment variable does not exist. It must be created pointing to the `build_system` repo directory in the local computer.
    goto end
)
if not exist %BUILD_SYSTEM_DIR%\ (
    echo ERROR: `BUILD_SYSTEM_DIR` environment variable does not point to a valid directory. It must be set to point to the `build_system` repo directory in the local computer.
    goto end
)

set VS_REPOS_DIR=%BUILD_SYSTEM_DIR:\=/%/..


rem INPUT PARAMETERS

set NUM_ARGS=0
for %%x in (%*) do set /A NUM_ARGS+=1
if %NUM_ARGS% == 0 (
    echo ERROR: The filename of the executable must be mandatorily provided as first argument
    goto end
) 

set EXE_FILE=%1
set EXE_COMMAND=%*

if not exist %EXE_FILE% (
    echo ERROR: Executable file "%EXE_FILE%" does not exist
)


rem FIND DEPENDENCIES REPOS

set THIRD_PARTY_REPO=%VS_REPOS_DIR%/third_party
if not exist %THIRD_PARTY_REPO%\ (
    echo ERROR: `third_party` repo does not exist at the specified location "%THIRD_PARTY_REPO%".
    goto end
)

set RELEASES_REPO=%VS_REPOS_DIR%/releases
if not exist %RELEASES_REPO%\ (
    echo ERROR: `releases` repo does not exist at the specified location "%RELEASES_REPO%".
    goto end
)


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


rem PROCESS DEPENDENCIES REPOS

echo SEARCHING DLL FILES IN THIRD PARTY REPO ("%THIRD_PARTY_REPO%")...
call :find_dll_subfolders %THIRD_PARTY_REPO%

echo SEARCHING DLL FILES IN RELEASES REPO ("%RELEASES_REPO%")...
call ::find_dll_subfolders %RELEASES_REPO%

goto run_command

:find_dll_subfolders
set COMMAND="find %~1 -xtype f -name '*.dll' | sed -r 's|/[^/]+$||'  | sort | uniq  | tr '/' '\\\\\\\\' > new_dependencies_paths.tmp"
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
echo RUNNING COMMAND "%EXE_COMMAND%" ...
%EXE_COMMAND%
goto end
 
:end
if exist new_dependencies_paths.tmp del new_dependencies_paths.tmp
endlocal