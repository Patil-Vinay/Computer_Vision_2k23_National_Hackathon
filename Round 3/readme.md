# Person Tracking with Multiple Overhead Cameras
Implement a robust person-tracking system using a network of overhead cameras to enhance security, optimize resource allocation, and gain valuable insights into crowd dynamics for applications like event management, retail analytics, and public safety.

## Key Functionalities

- Real-time tracking of individuals and groups.
- Accurate location and trajectory data.
- Automated alerts for security and safety incidents.
- Robust tracking under occlusions and varying lighting conditions.
- Data storage and analytics for post-location analysis.

## Technical Stack

**Programming Languages**
- Python

**Libraries/Frameworks**
- **PyTorch:** Used for deep learning-based object detection and tracking.
- **OpenCV (cv2):** Used for video input/output, image processing, and graphical user interface.
- **NumPy:** Used for numerical operations and array manipulation.
- **argparse:** Used for command-line argument parsing.
- **torchvision:** Part of PyTorch, used for computer vision tasks and pre-trained models.
- **time:** Used for measuring execution time and timing operations.

**Deep Learning Models**
- Faster R-CNN (ResNet50 FPN variant) for object detection. The model is loaded from Torchvision's models.
- DeepSORT: A deep learning-based object tracking algorithm.

**Data Handling**
- TorchVision transforms: Used for image resizing and transformation to tensors.
- ToTensor: Converts images to PyTorch tensors.

**Other Tools and Utilities**
- argparse: Used for command-line argument parsing.
- os: Used for file and directory operations.
- Random number generation (for assigning colors to object tracks).

**Video Processing**
- Video capture (using OpenCV) from two input sources (webcams or video files).
- VideoWriter (using OpenCV) for saving the processed video with annotations.

**Hardware Acceleration**
- Utilizes CUDA for GPU acceleration

**Command to Run:**
```bash
python3 deep.py --input1 5.mp4 --input2 6.mp4 --embedder mobilenet --model fasterrcnn_resnet50_fpn_v2
```

## Output
### Images-
![](https://github.com/Patil-Vinay/Computer_Vision_2k23_National_Hackathon/blob/main/Round%203/Crowd%20Management%20using%20Yolov8/Dashboard%20UI.png)

![](https://github.com/Patil-Vinay/Computer_Vision_2k23_National_Hackathon/blob/main/Round%203/Crowd%20Management%20using%20Yolov8/Security%20Dashboard%20UI.png)

![](https://github.com/Patil-Vinay/Computer_Vision_2k23_National_Hackathon/blob/main/Round%203/Crowd%20Management%20using%20Yolov8/Analytics%20UI.png)

### Video-
[![Watch the video](https://img.youtube.com/vi/AMzIEpD2tVQ/maxresdefault.jpg)](https://youtu.be/AMzIEpD2tVQ)
