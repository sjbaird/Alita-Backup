@echo off
setlocal

set "TARGET_DIR=C:\Alita"
set "SCRIPTS_DIR=%TARGET_DIR%\scripts"
set "OUTPUT_FILE=%TARGET_DIR%\folder_contents.txt"
set "APP_FILE=%TARGET_DIR%\app.py"
set "ERROR_LOG=%TARGET_DIR%\error_log.txt"

:: Delete the old output file if it exists
if exist %OUTPUT_FILE% del %OUTPUT_FILE%

:: Start with an explanation
echo This list is the directory structure and file contents for scripts that are relevant. >> %OUTPUT_FILE%
echo Each section is treated as an individual script for clarity. >> %OUTPUT_FILE%
echo. >> %OUTPUT_FILE%

:: List directory structure and file contents for specified directories and files
echo Listing directory structure and file contents for %SCRIPTS_DIR% and %APP_FILE% excluding __pycache__ and profiles folder. >> %OUTPUT_FILE%
echo. >> %OUTPUT_FILE%

echo ---Contents of app.py--- >> %OUTPUT_FILE%
if exist %APP_FILE% (
    echo *** START OF FILE: %APP_FILE% *** >> %OUTPUT_FILE%
    type "%APP_FILE%" >> %OUTPUT_FILE%
    echo *** END OF FILE: %APP_FILE% *** >> %OUTPUT_FILE%
) else (
    echo ERROR: %APP_FILE% not found. >> %ERROR_LOG%
)
echo. >> %OUTPUT_FILE%

echo ---Contents of Python scripts in TARGET_DIR--- >> %OUTPUT_FILE%
for %%i in ("%TARGET_DIR%\*.py") do (
    echo File: %%i >> %OUTPUT_FILE%
    type "%%i" >> %OUTPUT_FILE%
    echo -------------------- >> %OUTPUT_FILE%
    echo. >> %OUTPUT_FILE%
)
echo. >> %OUTPUT_FILE%

echo ---Contents of SCRIPTS_DIR--- >> %OUTPUT_FILE%
if exist %SCRIPTS_DIR% (
    echo *** START OF DIRECTORY: %SCRIPTS_DIR% *** >> %OUTPUT_FILE%
    for /f "delims=" %%i in ('dir /b /s /a-d %SCRIPTS_DIR%\*.py ^| findstr /v /i /c:"\profiles" /c:"__pycache__"') do (
        echo -------------------- >> %OUTPUT_FILE%
        echo File: %%i >> %OUTPUT_FILE%
        type "%%i" >> %OUTPUT_FILE%
        echo -------------------- >> %OUTPUT_FILE%
        echo. >> %OUTPUT_FILE%
    )
    echo *** END OF DIRECTORY: %SCRIPTS_DIR% *** >> %OUTPUT_FILE%
) else (
    echo ERROR: %SCRIPTS_DIR% not found. >> %ERROR_LOG%
)
echo. >> %OUTPUT_FILE%

echo Listing completed. >> %OUTPUT_FILE%

::python app.py
