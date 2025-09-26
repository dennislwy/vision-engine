import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import colors

logger = logging.getLogger(__name__)

try:
    from vision_engine.object_detectors.object_detector import ObjectDetector
except ImportError:
    from object_detectors.object_detector import ObjectDetector


class YOLOObjectDetector(ObjectDetector):
    """
    YOLO Object Detector class using ultralytics YOLO.
    """

    def __init__(
        self,
        model: Union[str, Path],
        device: Optional[str] = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        classes: Optional[list] = None,
    ):
        """
        Initialize the object detector class.

        Args:
            model: Path to the YOLO model weights file
            device: Device to run inference on ('cpu', 'cuda', etc.)
            conf: Confidence threshold for detections (default 0.25)
            iou: IoU threshold for non-maximum suppression (default 0.45)
            imgsz: Image size for inference (default 640)
            classes: The list of classes to detect (None for all classes)
        """
        self.model = YOLO(model, task="detect")
        self.device = self._parse_device(device)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self._classes = classes

        # Get class names from the model
        self._class_names = self.model.names

    def get_color(self, class_id: int) -> tuple[int, int, int]:
        """
        Get a color for a given class ID.

        Args:
            class_id: The class ID for which to get the color.

        Returns:
            A tuple representing the color (R, G, B).
        """
        return colors(class_id, True)

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
        # Run inference
        results = self.model(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=self._classes,
            device=self.device,
            verbose=False,
        )

        # Extract results from the first (and only) image
        result = results[0]

        bboxes = []
        scores = []
        class_ids = []  # Keep track of class IDs for color indexing

        if result.boxes is not None:
            # Get bounding boxes in xyxy format
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_indices = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf_score, class_id in zip(boxes, confidences, class_indices):
                xmin, ymin, xmax, ymax = box.astype(int)

                bboxes.append([xmin, ymin, xmax, ymax])
                scores.append(conf_score)
                class_ids.append(class_id)

        return (bboxes, class_ids, scores)

    @property
    def classes(self) -> Dict[int, str]:
        """
        Get the mapping of class IDs to class names.

        Returns:
            Dictionary mapping class IDs to class names.
        """
        return self._class_names

    def _parse_device(self, device: str = None):
        """
        Parse and validate the device string.

        Args:
            device: Device string ('cpu', 'cuda', 'cuda:0', etc.)

        Returns:
            Validated device string.
        """
        # Auto-select device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(
                    "CUDA detected. Using GPU: %s (%.1f GB)", gpu_name, gpu_memory
                )
            else:
                device = "cpu"
                logger.info("CUDA not available. Using CPU for inference.")

        # Validate device string
        else:
            if device.startswith("cuda"):
                if not torch.cuda.is_available():
                    logger.warning("CUDA not available. Falling back to CPU.")
                    device = "cpu"
                elif device != "cuda" and ":" in device:
                    try:
                        gpu_idx = int(device.split(":")[1])
                        if gpu_idx >= torch.cuda.device_count():
                            logger.info(
                                "GPU %d not available. Using default CUDA device.",
                                gpu_idx,
                            )
                            device = "cuda"
                        else:
                            # Show specific GPU info
                            gpu_name = torch.cuda.get_device_name(gpu_idx)
                            gpu_memory = (
                                torch.cuda.get_device_properties(gpu_idx).total_memory
                                / 1024**3
                            )
                            logger.info(
                                "Using GPU %d: %s (%.1f GB)",
                                gpu_idx,
                                gpu_name,
                                gpu_memory,
                            )
                    except (ValueError, IndexError):
                        logger.warning(
                            "Invalid CUDA device format. Using default CUDA device."
                        )
                        device = "cuda"
                else:
                    # Show default GPU info
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = (
                        torch.cuda.get_device_properties(0).total_memory / 1024**3
                    )
                    logger.info("Using GPU: %s (%.1f GB)", gpu_name, gpu_memory)
        return device
