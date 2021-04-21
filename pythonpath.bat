@ECHO OFF
setlocal
set PYTHONPATH=%PYTHONPATH%;.\src
python -m %*
endlocal

