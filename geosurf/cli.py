"""
CLI entry for GeoSurf.
This file simply re-exports the main() function from geosurf.transform.
"""
from .transform import main as main  # setuptools console_scripts target
