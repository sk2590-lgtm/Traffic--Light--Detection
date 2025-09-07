Traffic Light Detection System 🚦
Overview

This project implements a Traffic Light Detection System using computer vision techniques. The system can detect traffic lights in real-time from a video or webcam feed, display the FPS (frames per second), and maintain a count of how many times each light (red, yellow, green) appears.

It’s useful for traffic monitoring, autonomous vehicles, or smart city applications.

Features

Real-time traffic light detection

Counts the number of times each light appears

Displays FPS for performance monitoring

Works with webcam or video input

Requirements

Python 3.x

OpenCV (pip install opencv-python)

NumPy (pip install numpy)

Other dependencies (if any, e.g., TensorFlow or PyTorch if used)

Usage

Clone the repository:

git clone https://github.com/username/traffic-light-detection.git
cd traffic-light-detection


Install dependencies:

pip install -r requirements.txt


Run the detection script:

python traffic_light_detection.py


Optional: Adjust the input source:

Webcam: cv2.VideoCapture(0)

Video file: cv2.VideoCapture('video.mp4')

Output

Live video feed with traffic lights detected

FPS displayed on the video

Counts of each light (Red, Yellow, Green) updated in real-time

Future Improvements

Integrate with a web-based interface for live deployment

Support multi-camera setups

Add AI-based detection for improved accuracy in low-light conditions
