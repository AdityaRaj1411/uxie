@echo off
title Personal Second Brain Buddy

:: Get the directory of this script
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

:: Auto-detect virtual environment
:: Priority: project-local venv > project-local .venv > UXIE_VENV env var
if exist "%PROJECT_DIR%venv\Scripts\activate.bat" (
    echo Activating venv...
    call "%PROJECT_DIR%venv\Scripts\activate.bat"
) else if exist "%PROJECT_DIR%.venv\Scripts\activate.bat" (
    echo Activating .venv...
    call "%PROJECT_DIR%.venv\Scripts\activate.bat"
) else if defined UXIE_VENV (
    if exist "%UXIE_VENV%\Scripts\activate.bat" (
        echo Activating UXIE_VENV...
        call "%UXIE_VENV%\Scripts\activate.bat"
    ) else (
        echo [ERROR] UXIE_VENV is set to "%UXIE_VENV%" but no activate.bat found there.
        pause
        exit /b 1
    )
) else (
    echo [WARNING] No virtual environment found.
    echo.
    echo   Option 1: Create a local venv:
    echo     python -m venv venv
    echo     venv\Scripts\activate
    echo     pip install -r requirements.txt
    echo.
    echo   Option 2: Set UXIE_VENV to your existing env path:
    echo     set UXIE_VENV=D:\path\to\your\env
    echo.
    pause
    exit /b 1
)

:: Clear the screen and launch the CLI
cls
python src\main.py

:: Pause if the app crashes or exits so the window doesn't instantly close
echo.
pause
