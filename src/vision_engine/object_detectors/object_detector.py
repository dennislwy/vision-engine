from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np


class ObjectDetector(ABC):
    """Abstract interface for object detector implementations"""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Tuple[list, list, list]:
        """
        Run object detection on a single frame.

        Args:
            frame: The input image frame.

        Returns:
            Tuple of (bboxes, class_ids, scores)
            - bboxes: List of [xmin, ymin, xmax, ymax] coordinates
            - class_ids: List of class indices
            - scores: List of confidence scores
        """

    @property
    @abstractmethod
    def classes(self) -> Dict[int, str]:
        """
        Get the mapping of class IDs to class names.

        Returns:
            Dictionary mapping class IDs to class names.
        """
