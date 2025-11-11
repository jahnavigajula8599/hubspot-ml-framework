"""
Launch FastAPI server for model serving.

Usage:
    python run_api.py
"""
import sys
from pathlib import Path

import uvicorn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    print("Starting HubSpot ML API Server...")
    print("API Docs: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop\n")

    uvicorn.run(
        "ml_framework.serving.api:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
