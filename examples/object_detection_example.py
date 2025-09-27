import logging

import cv2

from vision_engine import (FrameGrabberService, GUIUtils, ManagedFrameGrabber,
                           ObjectDetectionService, YOLOObjectDetector)

logger = logging.getLogger(__name__)
gu = GUIUtils()

source = 0  # 0 for webcam, or replace with video file path or stream URL
model = "yolo11n.pt"

def main():
    # Setup frame grabbing
    grabber = ManagedFrameGrabber(source=source)
    frame_service = FrameGrabberService(grabber)
    frame_service.add_queue("display", max_size=1)
    frame_service.add_queue("detection", max_size=1)

    # Setup object detection
    detector = YOLOObjectDetector(model=model)
    detection_queue = frame_service.get_queue("detection")
    detection_service = ObjectDetectionService(
        detection_queue=detection_queue,
        object_detector=detector,
        buffer_size=1,
        calc_fps=True,
    )

    # Start services
    frame_service.start()
    detection_service.start()

    prev_result = result = None

    try:
        while True:
            # Get frame for display
            frame = frame_service.get_frame("display", timeout=0.1)
            if frame is not None:

                # Get detection results
                result = detection_service.get_latest_detection(timeout=0.1)
                if not result:
                    result = prev_result

                if result:
                    timestamp, detection_frame, (bboxes, class_ids, scores) = result
                    # Annotate frame with detection results
                    frame = gu.draw_detection_annotator(
                        frame, bboxes, class_ids, scores, detector.classes
                    )
                    prev_result = result

                # Display FPS on frame
                fps = detection_service.get_fps()
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("Video", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        detection_service.stop()
        frame_service.release()
        cv2.destroyAllWindows()


def setup_logging() -> None:
    """Configure logging system with detailed formatting.

    Sets up the logging system to provide comprehensive debugging information
    including timestamps, module names, function names, and log levels. This
    configuration is essential for monitoring the application's behavior and
    troubleshooting issues.

    The log format includes:
    - Timestamp: When the log entry was created
    - Logger name: Which module/class generated the log (truncated to 20 chars)
    - Function name: Which function generated the log (truncated to 20 chars)
    - Level: Log severity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Message: The actual log content

    Note:
        This replaces any existing logging handlers with the new configuration.
        All loggers will use DEBUG level, providing maximum detail.
    """
    # Set root logger to DEBUG level for maximum verbosity
    logging.basicConfig(level=logging.DEBUG)

    # Create detailed formatter for log messages
    formatter = logging.Formatter(
        "%(asctime)s [%(name)20.20s][%(funcName)20.20s][%(levelname)5.5s] %(message)s"
    )

    # Create console handler for output to terminal/console
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # Replace existing handlers with our configured handler
    logging.getLogger().handlers = [handler]


if __name__ == "__main__":
    # Configure logging before running main application
    setup_logging()

    # Start the main detection pipeline
    main()
