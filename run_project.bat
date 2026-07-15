@echo off
REM ============================================================================
REM  ZIMSEC O-Level Predictive Analytics — Project Launcher
REM ============================================================================
REM  Author: Cesario Machinga
REM
REM  This script:
REM    1. Verifies Python is installed
REM    2. Creates a virtual environment in .\venv (if missing)
REM    3. Installs dependencies from requirements.txt (only when needed)
REM    4. Verifies data files are in place
REM    5. Launches the FastAPI dashboard at http://127.0.0.1:8000
REM ============================================================================

title ZIMSEC O-Level Predictive Analytics System
color 0A
setlocal enabledelayedexpansion

echo.
echo ============================================================
echo   ZIMSEC O-Level Predictive Analytics System
echo   Predictive Modelling of District-Level Academic Performance
echo   Author: Cesario Machinga
echo ============================================================
echo.

REM ---- Move to the script's directory so all paths are relative ----
cd /d "%~dp0"

REM ---- Check Python is available ----
python --version >nul 2>&1
if errorlevel 1 (
    color 0C
    echo [ERROR] Python is not installed or not in PATH.
    echo.
    echo Please install Python 3.10 or newer from https://python.org
    echo Make sure to tick "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VERSION=%%v
echo [OK] Python !PY_VERSION! detected

REM ---- Verify requirements.txt is present ----
if not exist "requirements.txt" (
    color 0C
    echo [ERROR] requirements.txt not found in this folder.
    echo.
    echo Expected: %~dp0requirements.txt
    echo Make sure you extracted the full project bundle.
    echo.
    pause
    exit /b 1
)

REM ---- Create venv if missing ----
if not exist "venv\Scripts\activate.bat" (
    echo.
    echo [INFO] Creating virtual environment in .\venv ...
    python -m venv venv
    if errorlevel 1 (
        color 0E
        echo [WARN] Could not create venv. Falling back to system Python.
        echo        Some packages may install to your global environment.
        goto :install_deps
    )
    echo [OK] Virtual environment created
    set FRESH_VENV=1
) else (
    set FRESH_VENV=0
)

REM ---- Activate venv ----
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment activated
)

:install_deps
REM ---- Install/update dependencies from requirements.txt ----
echo.
echo [INFO] Checking dependencies (this can take a minute on first run)...

REM Upgrade pip itself first (avoids many install errors)
python -m pip install --upgrade pip --quiet --disable-pip-version-check >nul 2>&1

REM On a fresh venv, do a full install. Otherwise just verify everything resolves.
if "!FRESH_VENV!"=="1" (
    echo [INFO] Fresh environment - installing all packages...
    python -m pip install -r requirements.txt --disable-pip-version-check
    if errorlevel 1 (
        color 0E
        echo [WARN] Some packages failed to install. The system may still partially work.
        echo        See the error above for details. Common causes:
        echo          - No internet connection
        echo          - Corporate firewall blocking pypi.org
        echo          - Insufficient disk space
        echo.
        pause
    ) else (
        echo [OK] All dependencies installed
    )
) else (
    REM Quiet check — only re-install if anything is missing
    python -m pip install -r requirements.txt --quiet --disable-pip-version-check
    if errorlevel 1 (
        echo [WARN] Some dependencies could not be verified. Continuing anyway.
    ) else (
        echo [OK] Dependencies verified
    )
)

REM ---- Verify project data files ----
echo.
echo [INFO] Checking project files...

set MISSING=0
if not exist "data\zimsec_olevel_district_data.csv" (
    echo [WARN] data\zimsec_olevel_district_data.csv NOT FOUND
    set MISSING=1
)
if not exist "webapp\main.py" (
    echo [ERROR] webapp\main.py NOT FOUND
    set MISSING=2
)
if not exist "webapp\ingestion.py" (
    echo [WARN] webapp\ingestion.py not found - upload page may fail
)

if "!MISSING!"=="2" (
    color 0C
    echo.
    echo Cannot launch - core webapp files are missing.
    pause
    exit /b 1
)
if "!MISSING!"=="1" (
    echo.
    echo The dataset is missing. The dashboard will load but show no data
    echo until you place zimsec_olevel_district_data.csv in the data\ folder
    echo or upload it through the dashboard's Upload page.
    echo.
)

REM ---- Make sure the upload directory exists (multi-glob ingestion uses it) ----
if not exist "data\uploads" mkdir "data\uploads"
if not exist "data\raw"     mkdir "data\raw"
if not exist "output\models" mkdir "output\models"
if not exist "output\tables" mkdir "output\tables"
if not exist "output\figures" mkdir "output\figures"

REM ---- Launch ----
echo.
echo ============================================================
echo   LOGIN CREDENTIALS
echo   Email:    admin@zimsec.ac.zw
echo   Password: admin123
echo ============================================================
echo.
echo [INFO] Starting server at http://127.0.0.1:8000 ...
echo [INFO] Browser will open automatically in 2 seconds.
echo [INFO] Press Ctrl+C in this window to stop the server.
echo.

REM Open the browser after a short delay (the start command runs in parallel)
start "" /b cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8000"

REM Run the server in the foreground so logs are visible and Ctrl+C works
python webapp\main.py

REM ---- After server stops ----
echo.
echo ============================================================
echo   Server stopped.
echo ============================================================
pause
endlocal
