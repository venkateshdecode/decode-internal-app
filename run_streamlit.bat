@echo off
echo Starting Streamlit with compatibility fixes...
echo.

REM Disable Streamlit file watcher (PyTorch / timm fix)
set STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

REM Suppress OpenCV logging (headless-safe)
set OPENCV_LOG_LEVEL=SILENT

REM Optional PyTorch stability (safe on CPU)
set KMP_DUPLICATE_LIB_OK=TRUE

REM Run Streamlit
streamlit run app.py --server.fileWatcherType none

pause
