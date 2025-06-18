@echo off

REM Installing Python packages using uv
uv pip install numpy
uv pip install pandas
uv pip install streamlit

REM Show message about VS Code extensions
echo ================================
echo Install these VS Code extensions:
echo 1. Ruff
echo 2. Python Debugger
echo 3. Jupyter
echo 4. Cline
echo ================================
pause
