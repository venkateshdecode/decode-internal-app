@echo off
echo Starting Streamlit with PyTorch compatibility fix...
echo.

REM Set environment variable to disable file watching
set STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

REM Run streamlit with file watcher disabled
streamlit run app.py --server.fileWatcherType none

pause