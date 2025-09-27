"""
Vision Engine YOLO: A Python package for YOLO-based computer vision processing.

This package provides modular components for:
- Frame grabbing from various sources
- Object detection using YOLO models
- GUI utilities for visualization
"""

__version__ = "0.1.0"
__author__ = "Dennis Lee"
__email__ = "wylee2000@gmail.com"

from .frame_grabbers import (FrameGrabberService, ManagedFrameGrabber,
                             SingleFrameGrabber)
from .gui import GUIUtils
from .object_detectors import ObjectDetectionService, YOLOObjectDetector

# Define what gets imported with "from vision_engine import *"
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "FrameGrabberService",
    "ManagedFrameGrabber", 
    "SingleFrameGrabber",
    "ObjectDetectionService",
    "YOLOObjectDetector",
    "GUIUtils",
]