import torch
import torchvision
import cv2
import os
import time
import argparse
import numpy as np

from torchvision.transforms import ToTensor
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils import convert_detections, annotate
from coco_classes import COCO_91_CLASSES

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input1',  # Input for the first video
    required=True,
    help='path to the first input video',
)
parser.add_argument(
    '--input2',  # Input for the second video
    required=True,
    help='path to the second input video',
)
parser.add_argument(
    '--imgsz', 
    default=None,
    help='image resize, 640 will resize images to 640x640',
    type=int
)
parser.add_argument(
    '--model',
    default='fasterrcnn_resnet50_fpn_v2',
    help='model name',
    choices=[
        'fasterrcnn_resnet50_fpn_v2',
        'fasterrcnn_resnet50_fpn',
        'fasterrcnn_mobilenet_v3_large_fpn',
        'fasterrcnn_mobilenet_v3_large_320_fpn',
        'fcos_resnet50_fpn',
        'ssd300_vgg16',
        'ssdlite320_mobilenet_v3_large',
        'retinanet_resnet50_fpn',
        'retinanet_resnet50_fpn_v2'
    ]
)
parser.add_argument(
    '--threshold',
    default=0.8,
    help='score threshold to filter out detections',
    type=float
)
parser.add_argument(
    '--embedder',
    default='mobilenet',
    help='type of feature extractor to use',
    choices=[
        "mobilenet",
        "torchreid",
        "clip_RN50",
        "clip_RN101",
        "clip_RN50x4",
        "clip_RN50x16",
        "clip_ViT-B/32",
        "clip_ViT-B/16"
    ]
)
parser.add_argument(
    '--cls', 
    nargs='+',
    default=[1],
    help='which classes to track',
    type=int
)
args = parser.parse_args()

np.random.seed(42)

OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COLORS = np.random.randint(0, 255, size=(len(COCO_91_CLASSES), 3))

print(f"Tracking: {[COCO_91_CLASSES[idx] for idx in args.cls]}")
print(f"Detector: {args.model}")
print(f"Re-ID embedder: {args.embedder}")

# Load model.
model = getattr(torchvision.models.detection, args.model)(weights='DEFAULT')
# Set model to evaluation mode.
model.eval().to(device)

# Initialize SORT tracker objects for each video.
tracker1 = DeepSort(max_age=30, embedder=args.embedder)
tracker2 = DeepSort(max_age=30, embedder=args.embedder)

VIDEO_PATH1 = args.input1
cap1 = cv2.VideoCapture(0)
frame_width1 = int(cap1.get(3))
frame_height1 = int(cap1.get(4))
frame_fps1 = int(cap1.get(5))
frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
save_name1 = VIDEO_PATH1.split(os.path.sep)[-1].split('.')[0]
out1 = cv2.VideoWriter(
    f"{OUT_DIR}/{save_name1}_{args.model}_{args.embedder}.mp4", 
    cv2.VideoWriter_fourcc(*'mp4v'), frame_fps1, 
    (frame_width1, frame_height1)
)

VIDEO_PATH2 = args.input2
cap2 = cv2.VideoCapture(2)
frame_width2 = int(cap2.get(3))
frame_height2 = int(cap2.get(4))
frame_fps2 = int(cap2.get(5))
frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
save_name2 = VIDEO_PATH2.split(os.path.sep)[-1].split('.')[0]
out2 = cv2.VideoWriter(
    f"{OUT_DIR}/{save_name2}_{args.model}_{args.embedder}.mp4", 
    cv2.VideoWriter_fourcc(*'mp4v'), frame_fps2, 
    (frame_width2, frame_height2)
)

frame_count1 = 0
frame_count2 = 0

while True:
    start_time1 = time.time()
    start_time2 = time.time()
    args.imgsz = 256
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if ret1:
        # Process and display frame1
        if args.imgsz is not None:
            resized_frame1 = cv2.resize(
                cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB), 
                (args.imgsz, args.imgsz)
            )
        else:
            resized_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame_tensor1 = ToTensor()(resized_frame1).to(device)
        
        with torch.no_grad():
            detections1 = model([frame_tensor1])[0]
        
        detections1 = convert_detections(detections1, args.threshold, args.cls)
        tracks1 = tracker1.update_tracks(detections1, frame=frame1)
        
        fps1 = 1 / (time.time() - start_time1)
        frame_count1 += 1
        
        print(f"Video 1 - Frame {frame_count1}/{frames1}", 
              f"Detection FPS: {fps1:.1f}")
        
        if len(tracks1) > 0:
            frame1 = annotate(
                tracks1, 
                frame1, 
                resized_frame1,
                frame_width1,
                frame_height1,
                COLORS
            )
        
        cv2.putText(
            frame1,
            f"FPS: {fps1:.1f}",
            (int(20), int(40)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        out1.write(frame1)
        cv2.imshow("Video 1 Output", frame1)

    if ret2:
        # Process and display frame2
        if args.imgsz is not None:
            resized_frame2 = cv2.resize(
                cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB), 
                (args.imgsz, args.imgsz)
            )
        else:
            resized_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        frame_tensor2 = ToTensor()(resized_frame2).to(device)
        
        with torch.no_grad():
            detections2 = model([frame_tensor2])[0]
        
        detections2 = convert_detections(detections2, args.threshold, args.cls)
        tracks2 = tracker2.update_tracks(detections2, frame=frame2)
        
        fps2 = 1 / (time.time() - start_time2)
        frame_count2 += 1
        
        print(f"Video 2 - Frame {frame_count2}/{frames2}", 
              f"Detection FPS: {fps2:.1f}")
        
        if len(tracks2) > 0:
            frame2 = annotate(
                tracks2, 
                frame2, 
                resized_frame2,
                frame_width2,
                frame_height2,
                COLORS
            )
        
        cv2.putText(
            frame2,
            f"FPS: {fps2:.1f}",
            (int(20), int(40)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        out2.write(frame2)
        cv2.imshow("Video 2 Output", frame2)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap1.release()
cap2.release()
out1.release()
out2.release()
cv2.destroyAllWindows()
