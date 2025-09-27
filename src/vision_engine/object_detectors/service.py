import logging
import queue
import threading
import time
from typing import Optional, Tuple

import numpy as np

try:
    from vision_engine.object_detectors.yolo import YOLOObjectDetector
except ImportError:
    from object_detectors.yolo import YOLOObjectDetector

logger = logging.getLogger(__name__)


class ObjectDetectionService:
    """Service for performing object detection on frames in a separate thread.

    This service runs object detection in a background thread, consuming frames
    from an input queue and producing detection results in an output queue.
    It provides a thread-safe interface for real-time object detection with
    configurable buffering to prevent memory buildup.

    The service maintains a single detection thread that continuously processes
    frames and stores results with timestamps. Old results are automatically
    discarded when the buffer is full to maintain memory efficiency.

    Attributes:
        _detection_queue (queue.Queue): Input queue for frames to be processed.
        _detector (YOLOObjectDetector): Object detector instance.
        _running (bool): Flag indicating if the service is running.
        _detection_thread (Optional[threading.Thread]): Background detection thread.
        _result_queue_lock (threading.Lock): Lock for thread-safe result queue access.
        _result_queue (queue.Queue): Output queue for detection results.
    """

    FPS_CALC_WINDOW = 30  # Number of frames to average for FPS calculation

    def __init__(
        self,
        detection_queue: queue.Queue,
        object_detector: YOLOObjectDetector,
        buffer_size: int = 1,
    ) -> None:
        """Initialize the object detection service.

        Args:
            detection_queue (queue.Queue): Queue containing frames (np.ndarray)
                to be processed for object detection.
            object_detector (YOLOObjectDetector): Configured YOLO detector instance
                that will perform the actual object detection.
            buffer_size (int, optional): Maximum number of detection results to
                buffer in the output queue. Defaults to 1. When full, oldest
                results are discarded to prevent memory buildup.

        Note:
            The service starts in a stopped state. Call start() to begin processing.
        """

        # Input queue for object detection - frames are consumed from here
        self._detection_queue = detection_queue

        # Object detector instance - performs the actual detection work
        self._detector = object_detector

        # Thread management - controls the background detection loop
        self._running = False
        self._detection_thread: Optional[threading.Thread] = None

        # Result queue management - stores detection results with thread safety
        self._result_queue_lock = threading.Lock()
        self._result_queue = queue.Queue(maxsize=buffer_size)

    def start(self) -> None:
        """Start the object detection thread.

        Creates and starts a daemon thread that continuously processes frames
        from the detection queue. The thread will automatically terminate when
        the main program exits.

        Raises:
            RuntimeError: If the service is already running. Call stop() first
                before starting again.

        Note:
            The detection thread is created as a daemon thread, meaning it will
            not prevent the program from exiting.
        """
        if self._running:
            raise RuntimeError("Object detection service is already running")

        self._running = True

        # Start detection thread
        self._detection_thread = threading.Thread(
            target=self._detection_loop, daemon=True
        )
        self._detection_thread.start()

        logger.info("Object detection service started")

    def stop(self) -> None:
        """Stop the object detection thread and clean up resources.

        Gracefully shuts down the detection thread and clears any remaining
        results from the output queue. This method is idempotent - calling
        it multiple times is safe.

        Note:
            This method will wait up to 2 seconds for the detection thread
            to finish processing its current frame before forcibly terminating.
        """
        if not self._running:
            logger.warning("Object detection service is already stopped")
            return

        logger.info("Stopping object detection service...")

        # Signal thread to stop - thread will check this flag in its loop
        self._running = False

        # Wait for detection thread to finish current processing cycle
        if self._detection_thread and self._detection_thread.is_alive():
            self._detection_thread.join(timeout=2.0)

        # Clear any remaining results from the output queue to free memory
        with self._result_queue_lock:
            while not self._result_queue.empty():
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    break

        logger.info("Object detection service stopped")

    def is_running(self) -> bool:
        """Check if the service is currently running.

        Returns:
            True if running, False otherwise.
        """
        return self._running

    def get_fps(self) -> float:
        """Get current frames per second.

        Returns:
            Current FPS based on rolling average.
        """
        return self._last_fps
    
    def _detection_loop(self) -> None:
        """Main loop for object detection processing.

        This method runs in a separate thread and continuously:
        1. Retrieves frames from the input queue
        2. Performs object detection on each frame
        3. Stores results with timestamps in the output queue
        4. Manages buffer overflow by discarding old results

        The loop continues until the _running flag is set to False by stop().

        Note:
            This is a private method and should not be called directly.
            It's automatically invoked when start() is called.
        """
        while self._running:
            # Get next frame from input queue
            frame = self._get_frame(timeout=0.1)
            if frame is None:
                # No frame available, continue to next iteration
                # This prevents busy-waiting and allows clean
                continue

            # Perform object detection on the frame
            results = self._detector.detect(frame)

            timestamp = time.time()

            # Thread-safe addition to result queue with overflow management
            # add timestamp, frame & detection results to result queue
            with self._result_queue_lock:
                if self._result_queue.full():
                    try:
                        self._result_queue.get_nowait()  # Discard oldest result
                    except queue.Empty:
                        pass

                # Add new result: (timestamp, original_frame, detection_results)
                self._result_queue.put((timestamp, frame, results))

            # Update performance metrics
            if self._calc_fps:
                self._calculate_fps()

    def get_latest_detection(
        self, timeout: Optional[float] = None
    ) -> Optional[Tuple[float, np.ndarray, Tuple[list, list, list]]]:
        """Get the latest detection results from the result queue.

        Retrieves the most recent detection result, which includes the original
        frame, detection timestamp, and all detected objects with their properties.

        Args:
            timeout (Optional[float], optional): Maximum time to wait for a result.
                - None or 0: Non-blocking, returns immediately if queue is empty
                - >0: Blocking, waits up to timeout seconds for a result
                Defaults to None.

        Returns:
            Optional[Tuple[float, np.ndarray, Tuple[list, list, list]]]:
                Detection result tuple containing:
                - float: Unix timestamp when detection was performed
                - np.ndarray: Original frame that was processed (HxWxC format)
                - Tuple[list, list, list]: Detection results from YOLO:
                    - list: Bounding boxes as [x1, y1, x2, y2] coordinates
                    - list: Class IDs for each detected object
                    - list: Confidence scores for each detection (0.0-1.0)
                Returns None if no results available within timeout period.

        Note:
            This method is thread-safe and can be called from multiple threads.
            Results are consumed from the queue, so each result is only returned once.
        """
        with self._result_queue_lock:
            try:
                if not timeout:
                    # Non-blocking: return immediately if queue empty
                    return self._result_queue.get_nowait()

                # Blocking: wait up to timeout seconds for a result
                return self._result_queue.get(timeout=timeout)
            except queue.Empty:
                # No results available within the specified timeout
                return None

    def _get_frame(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """Get a frame from the specified queue.

        Args:
            queue_name: Name of queue to get frame from.
            timeout: Timeout in seconds (>0 = blocking, None or 0 = non-blocking).

        Returns:
            Frame as numpy array, or None if timeout/queue empty.

        Raises:
            KeyError: If queue name doesn't exist.
        """
        try:
            # Non-blocking Behavior:
            # - If an item is immediately available, it returns it.
            # - If the queue is empty, it raises queue.Empty immediately instead of waiting.
            if not timeout:
                return (
                    self._detection_queue.get_nowait()
                )  # non-blocking, equivalent q.get(block=False)

            # Blocking Behavior:
            # - If an item is available right away, it returns it.
            # - If the queue is empty, it waits up to timeout seconds for an item.
            # - If nothing appears before the timeout expires, it raises queue.Empty.
            return self._detection_queue.get(
                timeout=timeout
            )  # blocking get with timeout
        except queue.Empty:
            # Queue is empty - this is normal during non-blocking calls
            # or when no frames arrive within the timeout per
            return None

    def _calculate_fps(self) -> None:
        """Update FPS calculation with current timestamp."""
        self._fps_counter.append(time.time())

        if len(self._fps_counter) > 3:
            time_span = self._fps_counter[-1] - self._fps_counter[0]
            if time_span > 0:
                self._last_fps = len(self._fps_counter) / time_span
            else:
                print("Warning: Time span for FPS calculation is zero")