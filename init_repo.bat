@echo off
echo ============================================================
echo   Green Route AQI - Git Repository Setup
echo ============================================================
echo.

echo [1/4] Initializing Git repository...
git init
if errorlevel 1 (
    echo ERROR: Git initialization failed
    pause
    exit /b 1
)

echo.
echo [2/4] Adding all files to Git...
git add .
if errorlevel 1 (
    echo ERROR: Failed to add files
    pause
    exit /b 1
)

echo.
echo [3/4] Creating initial commit...
git commit -m "Initial commit: Green Route AQI Forecasting System

- Add comprehensive AQI forecasting with ARIMA, LSTM, TCN models
- Include California housing dataset adaptation
- Add visualization and analysis tools
- Complete project structure with documentation
- Ready for production use with 17,000+ data records"

if errorlevel 1 (
    echo ERROR: Failed to create commit
    pause
    exit /b 1
)

echo.
echo [4/4] Repository setup complete!
echo.
echo ============================================================
echo   NEXT STEPS:
echo ============================================================
echo.
echo 1. Go to https://github.com and create a new repository
echo    Repository name: green-route-aqi
echo    Description: Air Quality Aware Navigation System with Advanced Forecasting
echo.
echo 2. Run these commands (replace YOUR-USERNAME):
echo    git remote add origin https://github.com/YOUR-USERNAME/green-route-aqi.git
echo    git branch -M main
echo    git push -u origin main
echo.
echo 3. Your repository will be live at:
echo    https://github.com/YOUR-USERNAME/green-route-aqi
echo.
echo ============================================================
echo   REPOSITORY READY FOR UPLOAD!
echo ============================================================

pause
