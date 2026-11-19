"""
Backend FastAPI Entry Point for PathSnap Map Matching Pipeline
Directs to api.api_app for unified route matching, Bayesian inference, and HMM decoding.
"""
from api.api_app import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
