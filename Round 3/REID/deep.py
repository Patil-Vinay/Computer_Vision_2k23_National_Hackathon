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
    help='path to the first input video or a webcam index like 0',
)
parser.add_argument(
    '--input2',  # Input for the second video
    required=True,
    help='path to the second input video or a webcam index like 1',
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
parser.add_argument(
    '--no-show',
    action='store_true',
    help='do not open preview windows, only save the output videos'
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


def open_source(source):
    # a plain number means a webcam index
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {source}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    save_name = os.path.splitext(os.path.basename(source))[0]
    out = cv2.VideoWriter(
        f"{OUT_DIR}/{save_name}_{args.model}_{args.embedder}.mp4",
        cv2.VideoWriter_fourcc(*'mp4v'), fps,
        (width, height)
    )
    return cap, out, width, height, frames


def track_frame(frame, tracker, width, height):
    start_time = time.time()
    if args.imgsz is not None:
        resized_frame = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            (args.imgsz, args.imgsz)
        )
    else:
        resized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = ToTensor()(resized_frame).to(device)

    with torch.no_grad():
        detections = model([frame_tensor])[0]

    detections = convert_detections(detections, args.threshold, args.cls)
    # boxes are in resized frame coords, so the embedder gets the resized frame too
    tracks = tracker.update_tracks(
        detections, frame=cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR))
    fps = 1 / (time.time() - start_time)

    if len(tracks) > 0:
        frame = annotate(tracks, frame, resized_frame, width, height, COLORS)

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (int(20), int(40)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 0, 255),
        thickness=2,
        lineType=cv2.LINE_AA
    )
    return frame, fps


# One tracker per camera so the IDs don't mix.
tracker1 = DeepSort(max_age=30, embedder=args.embedder)
tracker2 = DeepSort(max_age=30, embedder=args.embedder)

cap1, out1, frame_width1, frame_height1, frames1 = open_source(args.input1)
cap2, out2, frame_width2, frame_height2, frames2 = open_source(args.input2)

frame_count1 = 0
frame_count2 = 0

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # stop once both videos have ended
    if not ret1 and not ret2:
        break

    if ret1:
        frame1, fps1 = track_frame(frame1, tracker1, frame_width1, frame_height1)
        frame_count1 += 1
        print(f"Video 1 - Frame {frame_count1}/{frames1}",
              f"Detection FPS: {fps1:.1f}")
        out1.write(frame1)
        if not args.no_show:
            cv2.imshow("Video 1 Output", frame1)

    if ret2:
        frame2, fps2 = track_frame(frame2, tracker2, frame_width2, frame_height2)
        frame_count2 += 1
        print(f"Video 2 - Frame {frame_count2}/{frames2}",
              f"Detection FPS: {fps2:.1f}")
        out2.write(frame2)
        if not args.no_show:
            cv2.imshow("Video 2 Output", frame2)

    if not args.no_show and cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap1.release()
cap2.release()
out1.release()
out2.release()
cv2.destroyAllWindows()
