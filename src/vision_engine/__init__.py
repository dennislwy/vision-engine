"""
Vision Engine YOLO: A Python package for YOLO-based computer vision processing.

This package provides a high-level interface for vision processing,
with support for images, videos, and real-time processing.
"""

__version__ = "0.1.0"
__author__ = "Dennis Lee"
__email__ = "wylee2000@gmail.com"

from .frame_grabbers import (FrameGrabberService, ManagedFrameGrabber,
                             SingleFrameGrabber)
from .gui import GUIUtils
from .object_detectors import ObjectDetectionService, YOLOObjectDetector

__all__ = [
    "FrameGrabberService",
    "ManagedFrameGrabber", 
    "SingleFrameGrabber",
    "ObjectDetectionService",
    "YOLOObjectDetector",
    "GUIUtils",
]