python -m venv venv
call venv\Scripts\activate
pip install --upgrade pip
pip install fastapi uvicorn[standard] scikit-learn pandas numpy joblib
python server.py
pause
