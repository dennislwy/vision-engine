import time
from collections import deque
from typing import Dict

import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator, colors


class GUIUtils:
    def __init__(self, fps_window=1.0):
        """Initialize GUIUtils with FPS calculation parameters.

        Args:
            fps_window (float): Time window in seconds for FPS calculation.
                Must be positive. Defaults to 1.0.

        Raises:
            ValueError: If fps_window is not positive.
        """
        if fps_window <= 0:
            raise ValueError("fps_window must be positive")

        self.fps_window = fps_window  # Time window in seconds for FPS calculation
        self.frame_times = deque()  # Queue to store frame timestamps efficiently
        self.fps = 0  # Current calculated FPS value

    def show_fps(self, frame: np.ndarray) -> np.ndarray:
        """Display FPS counter on the given frame with background overlay.

        Args:
            frame (np.ndarray): OpenCV frame (numpy array) to draw FPS on.
                Should be in BGR color format.

        Returns:
            np.ndarray: Modified frame with FPS counter overlay in top-left corner.
                Returns the same frame object with modifications applied.
        """
        # Record current frame time for FPS calculation
        current_time = time.time()
        self.frame_times.append(current_time)

        # Remove timestamps outside the time window to maintain sliding window
        while self.frame_times and current_time - self.frame_times[0] > self.fps_window:
            self.frame_times.popleft()

        # Calculate average FPS over the time window
        if len(self.frame_times) > 1:
            # Calculate time span between oldest and newest frame
            time_span = self.frame_times[-1] - self.frame_times[0]
            if time_span > 0:
                # FPS = number of frame intervals / time span
                self.fps = (len(self.frame_times) - 1) / time_span
            else:
                self.fps = 0  # Avoid division by zero
        else:
            self.fps = 0  # Not enough frames to calculate FPS

        # Create FPS display text
        fps_text = f"FPS: {self.fps:.1f}"

        # Text rendering properties for clear visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_color = (255, 255, 255)  # White text for visibility
        bg_color = (0, 0, 0)  # Black background for contrast

        # Get text dimensions for proper background sizing
        (text_width, text_height), baseline = cv2.getTextSize(
            fps_text, font, font_scale, thickness
        )

        # Position for top-left corner with padding
        x, y = 10, 30

        # Draw black background rectangle with padding around text
        cv2.rectangle(
            frame,
            (x - 5, y - text_height - 5),  # Top-left corner
            (x + text_width + 5, y + baseline + 5),  # Bottom-right corner
            bg_color,
            -1,  # Filled rectangle
        )

        # Draw white text on top of background
        cv2.putText(frame, fps_text, (x, y), font, font_scale, text_color, thickness)

        return frame

    def draw_detection_cv2(
        self,
        frame: np.ndarray,
        bboxes: list,
        class_ids: list,
        scores: list,
        class_names: Dict[int, str],
        line_width: int = 2,
    ) -> np.ndarray:
        """Draw object detection results on frame using OpenCV primitives.

        This method provides a pure OpenCV implementation for drawing bounding boxes
        and labels, offering more control over rendering details and potentially
        better performance than the Annotator-based method.

        Args:
            frame (np.ndarray): Input frame to draw detections on. Modified in-place.
            bboxes (list): List of bounding boxes in format [xmin, ymin, xmax, ymax].
                Each bbox should be a sequence of 4 numeric values.
            class_ids (list): List of class IDs corresponding to each detection.
                Should be integers matching keys in class_names dict.
            scores (list): List of confidence scores for each detection.
                Values should be between 0.0 and 1.0.
            class_names (Dict[int, str]): Mapping from class ID to class name.
                Used to display human-readable labels.
            line_width (int): Thickness of bounding box lines. Defaults to 2.

        Returns:
            np.ndarray: Frame with detection overlays drawn. Same object as input frame.
        """
        # Pre-calculate common text rendering values for efficiency
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_thickness = 1

        # Convert lists to numpy arrays for faster iteration (if not already)
        if not isinstance(bboxes, np.ndarray):
            bboxes = np.array(bboxes)
        if not isinstance(class_ids, np.ndarray):
            class_ids = np.array(class_ids)
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)

        # Draw each detection with bounding box and label
        for bbox, class_id, score in zip(bboxes, class_ids, scores):

            # Get human-readable class name from ID
            class_name = class_names[class_id]

            # Format label with class name and confidence percentage
            score_percent = int(score * 100)
            label_text = f"{class_name}: {score_percent}%"

            # Get consistent color for this class (from ultralytics color palette)
            color = colors(class_id, True)

            # Convert bbox coordinates to integers for pixel operations
            xmin, ymin, xmax, ymax = map(int, bbox)

            # Draw bounding box rectangle
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, line_width)

            # Calculate optimal label position above bounding box
            labelSize, baseLine = cv2.getTextSize(
                label_text, font, font_scale, text_thickness
            )
            # Ensure label doesn't go above frame top
            label_ymin = max(ymin, labelSize[1] + 10)

            # Draw filled background rectangle for label text
            cv2.rectangle(
                frame,
                (xmin, label_ymin - labelSize[1] - 10),  # Top-left
                (xmin + labelSize[0], label_ymin + baseLine - 10),  # Bottom-right
                color,
                cv2.FILLED,  # Fill the rectangle
            )

            # Draw black label text on colored background for contrast
            cv2.putText(
                frame,
                label_text,
                (xmin, label_ymin - 7),  # Slight vertical offset for better positioning
                font,
                font_scale,
                (0, 0, 0),  # Black text color
                text_thickness,
            )

        return frame

    def draw_detection_annotator(
        self,
        frame: np.ndarray,
        bboxes: list,
        class_ids: list,
        scores: list,
        class_names: Dict[int, str],
        line_width: int = 2,
    ) -> np.ndarray:
        """Draw object detection results using ultralytics Annotator.

        This method uses the ultralytics Annotator class for drawing detections,
        which provides consistent styling with YOLO model outputs and may include
        additional visual enhancements.

        Args:
            frame (np.ndarray): Input frame to draw detections on.
                Should be in BGR color format.
            bboxes (list): List of bounding boxes in format [xmin, ymin, xmax, ymax].
                Each bbox should be a sequence of 4 numeric values.
            class_ids (list): List of class IDs corresponding to each detection.
                Should be integers matching keys in class_names dict.
            scores (list): List of confidence scores for each detection.
                Values should be between 0.0 and 1.0.
            class_names (Dict[int, str]): Mapping from class ID to class name.
                Used to display human-readable labels.
            line_width (int): Thickness of bounding box lines. Defaults to 2.

        Returns:
            np.ndarray: New frame with detection overlays drawn.
                This is a copy of the input frame with annotations applied.
        """
        # Initialize ultralytics Annotator with specified line width
        annotator = Annotator(frame, line_width=line_width)

        # Draw each detection using the annotator's optimized methods
        for bbox, class_id, score in zip(bboxes, class_ids, scores):
            # Get human-readable class name from ID
            class_name = class_names[class_id]

            # Format label with class name and confidence percentage
            label_text = f"{class_name}: {int(score * 100)}%"

            # Draw bounding box with label using consistent ultralytics styling
            annotator.box_label(bbox, label_text, color=colors(class_id, True))

        # Return the annotated frame (creates a copy)
        return annotator.result()
