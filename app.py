"""
Root FastAPI application entrypoint.
Imports production application from src.serving.
"""

from src.serving import app

# For direct execution: python app.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
