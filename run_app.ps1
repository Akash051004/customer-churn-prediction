# PowerShell script to run the Streamlit app
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& ".\.venv\Scripts\Activate.ps1"
& ".\.venv\Scripts\python.exe" -m streamlit run app/stremlit_app.py