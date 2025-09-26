import cv2

from vision_engine import FrameGrabberService, ManagedFrameGrabber

# Create frame grabber for webcam
grabber = ManagedFrameGrabber(source=0)

# Create service with FPS limiting and monitoring
service = FrameGrabberService(
    frame_grabber=grabber,
    calc_fps=True
)

# Add a queue for frame distribution
service.add_queue("main", max_size=1)

# Start capturing
service.start()

try:
    while True:
        # Get latest frame
        frame = service.get_frame("main", timeout=0.1)
        if frame is not None:
            # Display FPS on frame
            fps = service.get_fps()
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Show frame
            cv2.imshow("Video", frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
finally:
    service.release()
    cv2.destroyAllWindows()