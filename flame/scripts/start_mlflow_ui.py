#!/usr/bin/env python3
"""
Script to start the MLflow UI server.

Usage:
    python scripts/start_mlflow_ui.py

Then open: http://localhost:5000
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Change to the project root directory
    project_root = Path(__file__).parent.parent
    
    print("Starting MLflow UI...")
    print("Once started, open: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    try:
        # Start MLflow UI server
        subprocess.run([
            sys.executable, "-m", "mlflow", "ui", 
            "--backend-store-uri", f"file://{project_root.absolute()}/mlruns"
        ], cwd=project_root)
    except KeyboardInterrupt:
        print("\nMLflow UI server stopped.")

if __name__ == "__main__":
    main()
