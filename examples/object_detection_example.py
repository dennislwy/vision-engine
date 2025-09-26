import cv2

from vision_engine import (FrameGrabberService, GUIUtils, ManagedFrameGrabber,
                           ObjectDetectionService, YOLOObjectDetector)

gui = GUIUtils()

# Setup frame grabbing
grabber = ManagedFrameGrabber(source=0)
frame_service = FrameGrabberService(grabber, calc_fps=True)
frame_service.add_queue("detection", max_size=2)
frame_service.start()

# Setup object detection
detection_queue = frame_service.get_queue("detection")
detector = YOLOObjectDetector(model="yolo11n.pt")
detection_service = ObjectDetectionService(
    detection_queue=detection_queue, object_detector=detector, buffer_size=1
)
detection_service.start()

try:
    while True:
        # Get detection results
        result = detection_service.get_latest_detection(timeout=0.1)
        if result:
            timestamp, frame, (bboxes, class_ids, scores) = result

            # Annotate frame with detection results
            annotated_frame = gui.draw_detection_annotator(
                frame, bboxes, class_ids, scores, detector.classes
            )

            # Show frame
            cv2.imshow("Video", frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


finally:
    detection_service.stop()
    frame_service.release()
    cv2.destroyAllWindows()
