import logging
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from vision_engine.frame_grabbers.frame_grabber import FrameGrabber
except ImportError:
    from frame_grabber import FrameGrabber


class ManagedFrameGrabber(FrameGrabber):
    """Frame grabber with automatic reconnection capabilities.

    This class extends the base FrameGrabber to provide automatic reconnection
    functionality when video capture fails. It implements exponential backoff
    for reconnection attempts and supports configurable retry limits.

    Args:
        source (Union[str, int]): Video source - can be a file path, URL, or
            device index.
        buffer_size (Optional[int], optional): OpenCV capture buffer size.
            Defaults to 1.
        enable_mjpg (bool, optional): Whether to enable MJPG codec for better
            USB camera performance. Defaults to True.
        auto_reconnect (bool, optional): Enable automatic reconnection on
            failures. Defaults to True.
        reconnect_delay (float, optional): Initial delay between reconnection
            attempts in seconds. Defaults to 1.0.
        reconnect_delay_max (float, optional): Maximum delay between
            reconnection attempts in seconds. Defaults to 30.0.
        max_reconnect_attempts (int, optional): Maximum number of reconnection
            attempts. Use -1 for unlimited attempts. Defaults to -1.

    Raises:
        ValueError: If any parameter validation fails.
    """

    def __init__(
        self,
        source: Union[str, int],
        buffer_size: Optional[int] = 1,
        enable_mjpg: bool = True,
        auto_reconnect: bool = True,
        reconnect_delay: float = 1.0,
        reconnect_delay_max: float = 30.0,
        max_reconnect_attempts: int = -1,
    ):
        # Input validation with descriptive error messages
        if buffer_size is not None and buffer_size < 1:
            raise ValueError("buffer_size must be > 0")
        if reconnect_delay < 0:
            raise ValueError("reconnect_delay must be >= 0")
        if reconnect_delay_max < reconnect_delay:
            raise ValueError("reconnect_delay_max must be >= reconnect_delay")
        if max_reconnect_attempts < -1:
            raise ValueError("max_reconnect_attempts must be >= -1")

        # Store configuration parameters for reconnection logic
        self.source = source
        self.buffer_size = buffer_size
        self.enable_mjpg = enable_mjpg
        self.auto_reconnect = auto_reconnect
        self.reconnect_delay = reconnect_delay
        self.reconnect_delay_max = reconnect_delay_max
        self.max_reconnect_attempts = max_reconnect_attempts

        # Initialize video capture object
        self.cap: Optional[cv2.VideoCapture] = None

        # Attempt initial connection to video source
        self._setup_capture()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the video source.

        Attempts to read a frame from the video capture. If the read fails
        and auto_reconnect is enabled, it will attempt to reconnect and
        try reading again.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: A tuple containing:
                - bool: True if frame was successfully read, False otherwise
                - Optional[np.ndarray]: The captured frame as a numpy array,
                  or None if capture failed
        """
        # Check if capture object is available
        if not self.cap:
            return False, None

        # Attempt to read frame from video source
        ret, frame = self.cap.read()

        # Handle read failure with automatic reconnection if enabled
        if not ret and self.auto_reconnect:
            logger.warning("Frame read failed")
            # Try to reconnect and read again
            if not self._attempt_reconnect():
                return False, None
            ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        """Release video capture resources.

        Properly releases the OpenCV VideoCapture object and sets the
        internal reference to None to prevent memory leaks.
        """
        if self.cap:
            logger.debug("Releasing video capture")
            self.cap.release()
            self.cap = None

    def isOpened(self) -> bool:
        """Check if the video capture is currently opened.

        Returns:
            bool: True if the video capture is opened and ready, False otherwise.
        """
        return self.cap.isOpened() if self.cap else False

    def __enter__(self):
        """Context manager entry.

        Returns:
            AutoReconnectFrameGrabber: Self reference for context management.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit.

        Ensures proper cleanup of resources when exiting the context.

        Args:
            exc_type: Exception type (if any).
            exc_val: Exception value (if any).
            exc_tb: Exception traceback (if any).
        """
        self.release()

    def _attempt_reconnect(self) -> bool:
        """Attempt to reconnect to the video source with exponential backoff.

        Implements an exponential backoff strategy for reconnection attempts.
        The delay between attempts increases exponentially until it reaches
        the maximum delay threshold.

        Returns:
            bool: True if reconnection was successful, False if all attempts
                failed or maximum attempts were reached.
        """
        attempt = 0

        # Continue attempting until max attempts reached or unlimited (-1)
        while attempt < self.max_reconnect_attempts or self.max_reconnect_attempts < 0:
            try:
                # Clean up existing connection before reconnecting
                self.release()

                # Calculate exponential backoff delay, capped at maximum
                delay = min(
                    self.reconnect_delay * (2**attempt), self.reconnect_delay_max
                )

                # Apply delay before reconnection attempt
                if delay > 0:
                    logger.info("Delaying %.1f seconds before reconnect...", delay)
                    time.sleep(delay)

                # Format attempt counter for logging
                attempt_str = (
                    f"{attempt + 1}/{self.max_reconnect_attempts}"
                    if self.max_reconnect_attempts > 0
                    else str(attempt + 1)
                )
                logger.info("Reconnecting attempt %s...", attempt_str)

                # Attempt to establish new connection
                if self._setup_capture():
                    logger.info("Reconnected successfully.")
                    return True

                # Log failed attempt with formatted counter
                logger.warning("Reconnect attempt %s failed.", attempt_str)

            except Exception as e:
                logger.error("Error during reconnect attempt: %s", e)

            attempt += 1

        return False

    def _setup_capture(self) -> bool:
        """Setup video capture with configured parameters.

        Initializes the OpenCV VideoCapture object with the specified source
        and applies configuration settings like buffer size and codec.

        Returns:
            bool: True if capture was successfully opened, False otherwise.
        """
        try:
            logger.info("Setting up video capture to source '%s'", self.source)

            # Create new VideoCapture instance
            self.cap = cv2.VideoCapture(self.source)

            # Configure buffer size if specified (reduces latency)
            if self.buffer_size is not None:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

            # Enable MJPG codec for improved USB camera performance
            if self.enable_mjpg:
                self.cap.set(
                    cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G")
                )

            # Verify that the capture was successfully opened
            return self.cap.isOpened()

        except Exception as e:
            logger.error("Error setting up video capture: %s", e)
            self.cap = None
            return False


if __name__ == "__main__":
    import sys

    def setup_logging():
        """Setup logging configuration for the script."""
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
    grabber = ManagedFrameGrabber(
        source=src, buffer_size=1, enable_mjpg=True, max_reconnect_attempts=3
    )
    logger.info("Capture started successfully.")

    # Performance timing setup
    start_time = time.time()
    frame_count = 0

    # Verify capture is ready before processing
    if not grabber.isOpened():
        logger.warning("Failed to open video source.")
        sys.exit(1)

    # Process frames for performance testing
    while frame_count < 30:
        _ret, _frame = grabber.read()
        if not _ret:
            logger.warning("Failed to capture frame.")
        frame_count += 1

    # Calculate and display performance metrics
    elapsed = time.time() - start_time
    grabber.release()
    logger.info("Capture released.")

    logger.info("Captured %d frames in %.2f seconds (%.2f FPS)", frame_count, elapsed, frame_count/elapsed)
