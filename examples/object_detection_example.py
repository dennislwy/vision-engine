import cv2

from vision_engine import (FrameGrabberService, GUIUtils, ManagedFrameGrabber,
                           ObjectDetectionService, YOLOObjectDetector)

gui = GUIUtils()

# Setup frame grabbing
grabber = ManagedFrameGrabber(source=0)
frame_service = FrameGrabberService(grabber, calc_fps=True)
frame_service.add_queue("display", max_size=1)
frame_service.add_queue("detection", max_size=1)

# Setup object detection
detection_queue = frame_service.get_queue("detection")
detector = YOLOObjectDetector(model="yolo11n.pt")
detection_service = ObjectDetectionService(
    detection_queue=detection_queue, object_detector=detector, buffer_size=1
)

# Start services
frame_service.start()
detection_service.start()

try:
    while True:
        # Get frame for display
        frame = frame_service.get_frame("display", timeout=0.1)
        if frame is not None:

            # Get detection results
            result = detection_service.get_latest_detection(timeout=0.1)
            if result:
                timestamp, detection_frame, (bboxes, class_ids, scores) = result

                # Annotate frame with detection results
                frame = gui.draw_detection_annotator(
                    frame, bboxes, class_ids, scores, detector.classes
                )

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
