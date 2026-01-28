@echo off
title ADX FPM - Passport Photo Processor
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo Starting ADX FPM server...
echo Open http://127.0.0.1:8080 in your browser
start http://127.0.0.1:8080
python run_web.py
pause
