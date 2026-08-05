@echo off
title Personal Second Brain Buddy (User Mode)

:: Get the directory of this script
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

:: Auto-detect virtual environment
:: Priority: project-local venv > project-local .venv > UXIE_VENV env var
if exist "%PROJECT_DIR%venv\Scripts\activate.bat" (
    call "%PROJECT_DIR%venv\Scripts\activate.bat"
) else if exist "%PROJECT_DIR%.venv\Scripts\activate.bat" (
    call "%PROJECT_DIR%.venv\Scripts\activate.bat"
) else if defined UXIE_VENV (
    if exist "%UXIE_VENV%\Scripts\activate.bat" (
        call "%UXIE_VENV%\Scripts\activate.bat"
    ) else (
        echo [ERROR] UXIE_VENV is set but no activate.bat found at "%UXIE_VENV%".
        pause
        exit /b 1
    )
) else (
    echo [ERROR] No virtual environment found. Set UXIE_VENV or create a local venv.
    pause
    exit /b 1
)

:: Set USER_MODE to suppress logs and show loading bar
set USER_MODE=1

:: Clear the screen and launch the CLI
cls
python src\main.py

:: Pause if the app crashes or exits so the window doesn't instantly close
echo.
pause
