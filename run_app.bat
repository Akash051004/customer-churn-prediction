@echo off
REM Activate virtual environment and run Streamlit app
call .\.venv\Scripts\activate.bat
streamlit run app/stremlit_app.py
pause