@ECHO OFF
setlocal
set PYTHONPATH=%PYTHONPATH%;.\src
python -m %1
endlocal

