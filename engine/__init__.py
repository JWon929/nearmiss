"""NearMiss analysis engine: calibration, trajectories, conflict metrics, privacy blur, validation.

Everything here is plain Python/NumPy/OpenCV (no web framework), so it can be unit-tested and read
top to bottom. The Streamlit pages in app.py only call into this package.
"""
__version__ = "0.2.0"
