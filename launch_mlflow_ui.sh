#!/bin/bash
# Launch MLflow UI to view experiments
# Usage: ./launch_mlflow_ui.sh [port]

PORT=${1:-5000}

echo "======================================"
echo "Starting MLflow UI"
echo "======================================"
echo ""
echo "MLflow tracking URI: ./mlruns"
echo "Port: $PORT"
echo ""
echo "Once started, open your browser to:"
echo "  http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop the server"
echo "======================================"
echo ""

mlflow ui --backend-store-uri ./mlruns --port $PORT
