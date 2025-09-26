import logging
from typing import Optional, Tuple, Union

import cv2
import numpy as np

try:
    from vision_engine.frame_grabbers.frame_grabber import FrameGrabber
except ImportError:
    from frame_grabber import FrameGrabber

logger = logging.getLogger(__name__)


class SingleFrameGrabber(FrameGrabber):
    """Frame grabber that captures one frame at a time with fresh connection.

    This implementation opens the video source, reads a single frame, and
    immediately closes the connection for each frame request. This approach
    ensures that only the latest frame is captured from the source, making
    it ideal for scenarios where:
    - Low latency access to the most recent frame is required
    - CPU usage needs to be minimized between captures
    - High frame rates are not necessary
    - Memory usage should be kept minimal

    Note: This approach may be slower than persistent connections due to
    the overhead of opening/closing the video stream for each frame.

    Args:
        source (Union[str, int]): Video source identifier. Can be:
            - String: File path or URL to video source
            - Integer: Camera device index (e.g., 0 for default camera)
    """

    def __init__(self, source: Union[str, int]):
        """Initialize the SingleFrameGrabber with a video source.

        Args:
            source (Union[str, int]): Video source identifier. Can be a file
                path, URL, or camera device index.
        """
        # Attempt to convert source to integer for camera index
        # If conversion fails, treat as string (file path or URL)
        try:
            src = int(source)
        except ValueError:
            src = source
        self._source = src

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the video source.

        Opens a fresh connection to the video source, captures one frame,
        and immediately closes the connection. This ensures the most recent
        frame is captured but introduces connection overhead.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: A tuple containing:
                - bool: True if frame was successfully captured, False otherwise
                - Optional[np.ndarray]: The captured frame as a numpy array,
                  or None if capture failed

        Raises:
            Exception: Any OpenCV-related exceptions are caught and logged,
                but not re-raised. Instead, returns (False, None).
        """
        cap: Optional[cv2.VideoCapture] = None

        try:
            # Create new VideoCapture instance for this frame request
            cap = cv2.VideoCapture(self._source)

            # Attempt to read frame only if capture opened successfully
            ret, frame = cap.read() if cap.isOpened() else (False, None)
            return ret, frame

        except Exception as e:
            # Log any errors that occur during frame capture
            logging.error("Error occurred while reading frame: %s", e)
            return False, None

        finally:
            # Always ensure VideoCapture is properly released
            if cap is not None:
                cap.release()
                cap = None

    def release(self) -> None:
        """Release video capture resources.

        For SingleFrameGrabber, this is a no-op since connections are
        not persistent. Each frame request creates and destroys its own
        VideoCapture instance.
        """
        pass

    def isOpened(self) -> bool:
        """Check if the video capture is currently opened.

        For SingleFrameGrabber, this always returns False since no
        persistent connection is maintained. Each frame request creates
        a temporary connection that is immediately closed.

        Returns:
            bool: Always False, as no persistent connection is maintained.
        """
        return False


if __name__ == "__main__":
    import sys
    import time

    def setup_logging():
        logging.basicConfig(level=logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [%(name)10.10s][%(funcName)20.20s][%(levelname)5.5s] %(message)s"
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logging.getLogger().handlers = [handler]

    setup_logging()

    # Read source from command line arguments, default to 0
    if len(sys.argv) > 1:
        try:
            src = int(sys.argv[1])
        except ValueError:
            src = sys.argv[1]  # Treat as string (file path or URL)
    else:
        src = 0  # webcam, change to your video source as needed

    logger.info("Starting capture from source: %s", src)
    grabber = SingleFrameGrabber(source=src)
    logger.info("Capture started successfully.")

    # Performance timing setup
    start_time = time.time()
    frame_count = 0

    # Process frames for performance testing
    while frame_count < 5:
        ret, frame = grabber.read()
        if not ret:
            logger.warning("Failed to capture frame.")
        frame_count += 1
        time.sleep(0.01)  # Simulate processing delay

    # Calculate and display performance metrics
    elapsed = time.time() - start_time
    grabber.release()
    logger.info("Capture released.")

    logger.info(
        f"Captured {frame_count} frames in {elapsed:.2f} seconds "
        f"({frame_count/elapsed:.2f} FPS)"
    )
