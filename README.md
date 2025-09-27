# Vision Engine

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful computer vision package for frame grabbing, object detection, and real-time video processing. Vision Engine provides thread-safe services for capturing frames from various sources and performing YOLO-based object detection with high performance and reliability.

## Features

- **Multi-threaded Frame Grabbing**: Capture frames from cameras, video files, or streams in background threads
- **Object Detection**: YOLO-based object detection with configurable models
- **Queue-based Architecture**: Thread-safe frame distribution and processing
- **Performance Monitoring**: Built-in FPS calculation and performance metrics
- **Flexible Configuration**: Customizable frame rates, sampling intervals, and buffer sizes
- **Robust Error Handling**: Automatic reconnection and graceful degradation
- **Easy Integration**: Simple API for embedding in larger applications

## Installation

### Install from Git Repository

You can install Vision Engine directly from the GitHub repository using pip:

```bash
# Install latest version from main branch
pip install git+https://github.com/dennislwy/vision-engine.git

# Install latest version from develop branch
pip install git+https://github.com/dennislwy/vision-engine.git@develop

# Install specific version/tag
pip install git+https://github.com/dennislwy/vision-engine.git@v0.1.0

# Install in development mode (editable)
pip install -e git+https://github.com/dennislwy/vision-engine.git#egg=vision-engine

```
**Using Poetry:**
```bash
# Install latest version from main branch
poetry add git+https://github.com/dennislwy/vision-engine.git

# Install latest version from develop branch
poetry add git+https://github.com/dennislwy/vision-engine.git@develop

# Install specific version/tag
poetry add git+https://github.com/dennislwy/vision-engine.git@v0.1.0

# Install in development mode (editable)
poetry add --editable git+https://github.com/dennislwy/vision-engine.git
```

### Requirements

- Python 3.9 or higher
- OpenCV Python (>=4.12.0.88)
- Ultralytics (>=8.3.203)

## Quick Start

### Basic Frame Grabbing

```python
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
```

### Object Detection Pipeline

```python
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
```

## API Reference

### Frame Grabbers

#### `FrameGrabberService`

A service for capturing frames in a background thread and distributing them to multiple queues.

**Constructor Parameters:**
- `frame_grabber`: FrameGrabber instance
- `target_fps`: Target frames per second (0 = unlimited)
- `queue_frame_skip`: Number of frames to skip between enqueues
- `queue_sample_interval`: Minimum seconds between enqueues
- `calc_fps`: Enable FPS calculation

**Key Methods:**
- `start()`: Start frame capture thread
- `stop()`: Stop capture and cleanup
- `add_queue(name, max_size)`: Add named queue for frame distribution
- `get_frame(queue_name, timeout)`: Get frame from specific queue
- `get_fps()`: Get current FPS

### Object Detection

#### `ObjectDetectionService`

A service for performing object detection on frames in a separate thread.

**Constructor Parameters:**
- `detection_queue`: Input queue containing frames
- `object_detector`: YOLOObjectDetector instance
- `buffer_size`: Maximum results to buffer

**Key Methods:**
- `start()`: Start detection thread
- `stop()`: Stop detection and cleanup
- `get_latest_detection(timeout)`: Get latest detection results

## Configuration Examples

### High-Performance Setup

```python
# Maximum speed capture with minimal buffering
service = FrameGrabberService(
    frame_grabber=grabber,
    target_fps=0,  # Unlimited FPS
    calc_fps=True
)
service.add_queue("main", max_size=1)  # Keep only latest frame
```

### Bandwidth-Limited Setup

```python
# Reduced frame rate for network streams
service = FrameGrabberService(
    frame_grabber=grabber,
    target_fps=5,  # 5 FPS
    queue_sample_interval=0.5,  # Max 2 frames per second to queues
    calc_fps=True
)
```

### Multi-Consumer Setup

```python
# Multiple processing streams
service.add_queue("display", max_size=1)      # UI display
service.add_queue("detection", max_size=3)    # Object detection
service.add_queue("recording", max_size=10)   # Video recording
```

## Error Handling

Vision Engine includes robust error handling:

- **Automatic Reconnection**: ManagedFrameGrabber can automatically reconnect to sources
- **Graceful Degradation**: Services continue running even if individual frames fail
- **Resource Cleanup**: Proper cleanup of threads and resources on shutdown
- **Queue Management**: Automatic overflow handling prevents memory leaks

## Performance Tips

1. **Choose Appropriate Buffer Sizes**: Larger buffers use more memory but provide smoother playback
2. **Monitor FPS**: Use `calc_fps=True` to monitor performance
3. **Adjust Target FPS**: Set realistic target FPS based on your processing capabilities
4. **Use Frame Skipping**: Skip frames for non-critical processing to reduce CPU load

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Dennis Lee - [wylee2000@gmail.com](mailto:wylee2000@gmail.com)

## Repository

[https://github.com/dennislwy/vision-engine](https://github.com/dennislwy/vision-engine)
