@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "TEX_FILE=physics_conventions_report.tex"
set "PDF_FILE=physics_conventions_report.pdf"
set "PDFLATEX=pdflatex"

where pdflatex >nul 2>nul
if errorlevel 1 (
    if exist "%LocalAppData%\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" (
        set "PDFLATEX=%LocalAppData%\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
    ) else (
        echo Could not find pdflatex on PATH or in the default MiKTeX install location.
        echo Install MiKTeX or add pdflatex to PATH, then try again.
        pause
        exit /b 1
    )
)

pushd "%SCRIPT_DIR%"
if errorlevel 1 (
    echo Could not enter the report directory:
    echo %SCRIPT_DIR%
    pause
    exit /b 1
)

echo Compiling physics_conventions_report.tex...
"%PDFLATEX%" -interaction=nonstopmode -halt-on-error "%TEX_FILE%"
if errorlevel 1 (
    popd
    echo.
    echo Build failed on pass 1. See physics_conventions_report.log for details.
    pause
    exit /b 1
)

"%PDFLATEX%" -interaction=nonstopmode -halt-on-error "%TEX_FILE%"
if errorlevel 1 (
    popd
    echo.
    echo Build failed on pass 2. See physics_conventions_report.log for details.
    pause
    exit /b 1
)

popd

echo.
echo PDF created successfully:
echo %SCRIPT_DIR%%PDF_FILE%
pause