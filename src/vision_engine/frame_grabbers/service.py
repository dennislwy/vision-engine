import logging
import queue
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

try:
    from vision_engine.frame_grabbers.frame_grabber import FrameGrabber
except ImportError:
    from frame_grabber import FrameGrabber

logger = logging.getLogger(__name__)


class FrameGrabberService:
    """Service for capturing frames in a background thread and distributing them to queues.

    This service continuously captures frames from a provided `FrameGrabber` in a
    separate daemon thread, and distributes copies of each frame to one or more
    named queues. It provides a thread-safe interface with optional frame-rate
    limiting, frame sampling/skip controls, and rolling FPS calculation.

    The service maintains a single capture thread that reads frames, applies
    time-based or count-based sampling (if configured), and enqueues frames into
    per-consumer queues. For bounded queues, the oldest frames are replaced to
    prevent memory buildup.

    Attributes:
        _frame_grabber (FrameGrabber): The underlying frame source.
        _target_fps (float): Target frames per second (0 = as fast as possible).
        _queue_frame_skip (int): Number of frames to skip between enqueues.
        _queue_sample_interval (float): Minimum seconds between enqueues.
        _running (bool): Flag indicating if the service is running.
        _capture_thread (Optional[threading.Thread]): Background capture thread.
        _capture_lock (threading.Lock): Lock protecting capture operations.
        _queues (dict[str, queue.Queue]): Mapping of queue names to queues.
        _queue_lock (threading.Lock): Lock protecting queue access and updates.
        _calc_fps (bool): Whether FPS calculation is enabled.
        _fps_counter (collections.deque): Rolling timestamps for FPS calc.
        _last_fps (float): Most recently computed FPS value.
        _last_frame_enqueue_time (float): Timestamp of the last enqueue event.
    """

    FPS_CALC_WINDOW = 30  # Number of frames to average for FPS calculation

    def __init__(
        self,
        frame_grabber: FrameGrabber,
        target_fps: Optional[float] = None,
        queue_frame_skip: Optional[int] = None,
        queue_sample_interval: Optional[float] = None,
        calc_fps: bool = False,
    ):
        """
        Frame grabber service that captures frames in a separate thread and
        distributes them to multiple named queues.

        Args:
            frame_grabber (FrameGrabber): An instance of a FrameGrabber subclass.
            target_fps (Optional[float]): Target frames per second for capturing
                frames. If None or 0, captures as fast as possible.
            queue_frame_skip (Optional[int]): Number of frames to skip before adding a
                frame to each queue. None or 0 means every frame is added.
            queue_sample_interval (Optional[float]): Minimum time interval in
                seconds between adding frames to each queue. If None or 0, no time
                interval is enforced.
            calc_fps (bool): Enable FPS calculation (default: False). If enabled,
                the service will maintain an FPS counter that can be queried.
        """
        # Input validation
        if target_fps is not None and target_fps < 0:
            raise ValueError("target_fps must be >= 0 or None")
        if queue_frame_skip is not None and queue_frame_skip < 0:
            raise ValueError("queue_frame_skip must be >= 0 or None")
        if queue_sample_interval is not None and queue_sample_interval < 0:
            raise ValueError("queue_sample_interval must be >= 0 or None")

        self._frame_grabber = frame_grabber
        self._target_fps = target_fps if target_fps is not None else 0
        self._queue_frame_skip = queue_frame_skip if queue_frame_skip is not None else 0
        self._queue_sample_interval = (
            queue_sample_interval if queue_sample_interval is not None else 0
        )

        # Thread management
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._capture_lock = threading.Lock()

        # Queue management - stores name -> queue mapping
        self._queues: dict[str, queue.Queue] = {}
        self._queue_lock = threading.Lock()

        # Performance monitoring
        self._calc_fps = calc_fps
        self._fps_counter: deque = deque(maxlen=self.FPS_CALC_WINDOW)
        self._last_fps = 0.0

        # Frame addition logic control
        self._last_frame_enqueue_time = 0.0

    def start(self) -> None:
        """Start the frame capture thread.

        Raises:
            RuntimeError: If already running or cannot start thread.
        """
        if self._running:
            raise RuntimeError("Frame grabber service is already running")

        self._running = True

        # Start capture thread
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        logger.info("Frame grabber service started")

    def stop(self) -> None:
        """Stop the frame capture thread and clean up resources."""
        if not self._running:
            logger.warning("Frame grabber service is already stopped")
            return

        logger.info("Stopping frame grabber service...")
        self._running = False

        # Wait for capture thread to finish
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)

        # Clear all queues
        with self._queue_lock:
            for q in self._queues.values():
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break

        logger.info("Frame grabber service stopped")

    def release(self) -> None:
        """Release resources and stop the grabber."""
        self.stop()
        self._frame_grabber.release()

    def add_queue(self, name: str, max_size: int = 1) -> None:
        """Add a named queue for frame distribution.

        Args:
            name: Unique name for the queue.
            max_size: Maximum queue size (0 = unlimited, 1 = latest frame only).

        Raises:
            ValueError: If queue name already exists.
        """
        with self._queue_lock:
            if name in self._queues:
                raise ValueError(f"Queue '{name}' already exists")
            self._queues[name] = queue.Queue(maxsize=max_size)

    def remove_queue(self, name: str) -> None:
        """Remove a named queue.

        Args:
            name: Name of queue to remove.

        Raises:
            KeyError: If queue name doesn't exist.
        """
        with self._queue_lock:
            if name not in self._queues:
                raise KeyError(f"Queue '{name}' not found")
            # Clear queue before removal
            q = self._queues[name]
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
            del self._queues[name]

    def get_queue(self, name: str) -> queue.Queue:
        """Get a reference to a named queue.

        Args:
            name: Name of queue to retrieve.

        Returns:
            The requested queue.

        Raises:
            KeyError: If queue name doesn't exist.
        """
        with self._queue_lock:
            if name not in self._queues:
                raise KeyError(f"Queue '{name}' not found")
            return self._queues[name]

    def get_frame(
        self, queue_name: str, timeout: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """Get a frame from the specified queue.

        Args:
            queue_name: Name of queue to get frame from.
            timeout: Timeout in seconds (>0 = blocking, None or 0 = non-blocking).

        Returns:
            Frame as numpy array, or None if timeout/queue empty.

        Raises:
            KeyError: If queue name doesn't exist.
        """
        q = self._queues[queue_name]

        try:
            # Non-blocking Behavior:
            # - If an item is immediately available, it returns it.
            # - If the queue is empty, it raises queue.Empty immediately instead of waiting.
            if not timeout:
                return q.get_nowait()  # non-blocking, equivalent q.get(block=False)

            # Blocking Behavior:
            # - If an item is available right away, it returns it.
            # - If the queue is empty, it waits up to timeout seconds for an item.
            # - If nothing appears before the timeout expires, it raises queue.Empty.
            return q.get(timeout=timeout)  # blocking get with timeout
        except queue.Empty:
            # print("Queue is still empty!")
            return None

    def is_running(self) -> bool:
        """Check if the frame grabber is currently running.

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

    def _capture_loop(self) -> None:
        """Main capture loop running in separate thread."""
        skip_counter = 0  # Counts frames to skip

        last_frame_enqueue_time = 0.0  # for Time-based frame addition

        # for FPS limiting
        last_frame_read_time = 0.0
        if self._target_fps:
            time_per_frame = 1.0 / self._target_fps

        while self._running:
            # FPS limiting
            if self._target_fps:
                remaining_time = max(
                    0, time_per_frame - (time.time() - last_frame_read_time)
                )
                logger.debug("Sleeping for %.2f seconds to control FPS", remaining_time)
                time.sleep(remaining_time)

            # Update last read time for FPS limiting
            last_frame_read_time = time.time()

            # Read frame from grabber
            _, frame = self._frame_grabber.read()
            if frame is None:
                print("No frame captured.")
                continue

            # Frame addition control
            if self._queue_sample_interval:
                # Time-based frame addition logic
                current_time = time.time()
                if current_time - last_frame_enqueue_time < self._queue_sample_interval:
                    continue
                last_frame_enqueue_time = current_time
            elif self._queue_frame_skip:
                # Frame skipping addition logic
                if skip_counter < self._queue_frame_skip:
                    skip_counter += 1
                    continue
                skip_counter = 0

            # Distribute frame to queues
            self._distribute_frame(frame)

            # Update performance metrics
            if self._calc_fps:
                self._calculate_fps()

    def _distribute_frame(self, frame: np.ndarray) -> None:
        """Distribute frame to all queues"""

        # Distribute to queues
        with self._queue_lock:
            for _, q in self._queues.items():
                frame_copy = frame.copy()  # Independent copy for each queue
                try:
                    if q.maxsize == 0:
                        # Unlimited queue - blocking put
                        q.put(frame_copy)
                    else:
                        # Limited queue - non-blocking put with replacement
                        q.put(frame_copy, block=False)
                except queue.Full:
                    # Replace oldest frame for limited queues
                    try:
                        q.get_nowait()
                        q.put(frame_copy, block=False)
                    except queue.Empty:
                        pass

    def _calculate_fps(self) -> None:
        """Update FPS calculation with current timestamp."""
        self._fps_counter.append(time.time())

        if len(self._fps_counter) > 3:
            time_span = self._fps_counter[-1] - self._fps_counter[0]
            if time_span > 0:
                self._last_fps = len(self._fps_counter) / time_span
            else:
                print("Warning: Time span for FPS calculation is zero")


if __name__ == "__main__":
    import sys

    import cv2

    try:
        from vision_engine.frame_grabbers import (ManagedFrameGrabber,
                                                  SingleFrameGrabber)
    except ImportError:
        from managed_frame_grabber import ManagedFrameGrabber
        from single_frame_grabber import SingleFrameGrabber

    def setup_logging():
        """Setup logging"""
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

    # grabber = ManagedFrameGrabber(source=src, max_reconnect_attempts=3)
    grabber = SingleFrameGrabber(source=src)

    service = FrameGrabberService(
        frame_grabber=grabber,
        target_fps=0,  # 0 = max speed, 1/4 = 4 seconds per frame
        queue_frame_skip=None,  # 0 = no skip, 1 = skip 1 frames between adds
        queue_sample_interval=0,  # 0 = no interval, 4.0 = 4 seconds between frames
        calc_fps=True,
    )

    service.add_queue("main", max_size=1)
    service.start()

    show = True  # Set to True to display frames
    snapshot = False  # Save frames to disk (yyyyMMdd_HHMMSS.jpg)

    run_seconds = 20

    logger.info("Demo will run for %d seconds", run_seconds)

    start_time = time.time()
    while time.time() - start_time < run_seconds:
        _frame = service.get_frame("main", timeout=0.1)

        if _frame is not None:
            if show:
                cv2.imshow("Frame", _frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if snapshot:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"snapshot_{timestamp}.jpg"
                cv2.imwrite(filename, _frame)
                logger.info("Saved snapshot to %s", filename)
        else:
            logger.warning("No frame received.")

    logger.info("Final FPS: %.2f", service.get_fps())
    service.stop()
    service.release()

    cv2.destroyAllWindows()
