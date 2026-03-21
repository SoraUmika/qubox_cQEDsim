@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_agent_workflow.ps1" %*
exit /b %ERRORLEVEL%
