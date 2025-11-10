@echo off
REM --- 1) Kích hoạt đúng env 'yolo' bằng activate.bat tuyệt đối ---
call "C:\Users\daong\anaconda3\Scripts\activate.bat" yolo

REM --- 2) PATH sạch: chỉ ưu tiên DLL/EXE trong env đang chạy ---
set "PATH=%CONDA_PREFIX%\Library\bin;%CONDA_PREFIX%\bin;%CONDA_PREFIX%\Scripts;C:\Windows\System32"

REM --- 3) Chạy python.exe của env yolo (không phụ thuộc PATH ngoài) ---
set "PYEXE=%CONDA_PREFIX%\python.exe"
echo Using Python: %PYEXE%
"%PYEXE%" "C:\ZaloAI2025\observing\train\yolov8_\main.py"
