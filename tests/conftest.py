"""Pytest configuration and shared fixtures."""
import sys
from pathlib import Path

# Add src to path so tests can import from ml_framework
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))