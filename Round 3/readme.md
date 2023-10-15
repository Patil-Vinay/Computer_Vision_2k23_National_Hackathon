# Round 3 - Person Tracking with Multiple Overhead Cameras

Problem statement: track people across a set of overhead cameras for security and crowd management.

This folder has three parts.

| Folder / file | What it does |
| --- | --- |
| `Crowd Management using Yolov8/` | Web app: a Flask home page plus two Streamlit dashboards for live person detection, counting and crowd alerts |
| `Crowd Management using Yolov8/realtime_stitching.py` | Stitches two camera feeds into one panorama and marks moving areas |
| `REID/deep.py` | Tracks people in two cameras at once with a torchvision detector and DeepSORT re-identification |

## 1. Crowd management app (YOLOv8)

- `app.py` is a small Flask site (home and about page) that links to the two dashboards.
- `streamlit_app.py` is the **Security Dashboard**
  - YOLOv8n detection or YOLOv8n-seg segmentation
  - source can be an uploaded image, one of the stored videos in `videos/` or the webcams
  - optional tracking with ByteTrack or BoT-SORT, with the path of each track drawn on the frame
  - counts the people in the frame and shows an alert when the count goes over the crowd limit set in the sidebar
  - webcam mode finds the connected cameras, runs them together (one model per camera so the track IDs don't mix) and uses the total of all cameras as the count
  - in video and webcam mode the count is logged with a timestamp to `count_data.csv`
- `crowd_analytics.py` is the **Analytics Dashboard**. It reads `count_data.csv`, redraws the line graph every 5 seconds and can make a short report with the highest and lowest count and when they happened.

The weights are in `weights/` (`yolov8n.pt`, `yolov8n-seg.pt`). If they are missing, ultralytics downloads them on first run.

### Run

```bash
cd "Round 3/Crowd Management using Yolov8"
pip install -r requirements.txt

python app.py                                              # http://127.0.0.1:5000
streamlit run streamlit_app.py --server.port 8501          # Security Dashboard
streamlit run crowd_analytics.py --server.port 8502        # Analytics Dashboard
```

Run the commands from inside the folder since the images, videos and logos are loaded with relative paths. The nav bar on the Flask page expects the dashboards on ports 8501 and 8502.

## 2. Panorama stitching

`realtime_stitching.py` reads webcams 0 and 1, matches SIFT keypoints between them, finds a homography with RANSAC and warps the right frame onto the left one. The homography is worked out once and then reused for every frame. A running average background (`pyimagesearch/basicmotiondetector.py`) marks the moving areas on the panorama.

```bash
python realtime_stitching.py
```

The cameras need to overlap and the frames have to be given left to right. Press `q` to quit.

## 3. Two camera tracking (REID)

`deep.py` runs a torchvision detector (Faster R-CNN ResNet50 FPN v2 by default) on two videos or webcams. People are passed to a DeepSORT tracker (`deep-sort-realtime`) that uses an appearance embedder (MobileNet by default) to keep the same ID on a person through occlusions. Each camera has its own tracker. The annotated videos are saved in `outputs/`.

```bash
cd "Round 3/REID"
pip install -r requirements.txt

python deep.py --input1 5.mp4 --input2 6.mp4 --embedder mobilenet --model fasterrcnn_resnet50_fpn_v2
python deep.py --input1 0 --input2 1 --imgsz 640      # two webcams
```

Useful options:

- `--imgsz 640` resizes frames before detection (faster on CPU)
- `--threshold 0.8` detection score threshold
- `--cls 1` COCO class ids to track (1 is person)
- `--no-show` only save the videos without opening windows

The `torchreid` and `clip_*` embedders need extra packages, see the [deep-sort-realtime](https://github.com/levan92/deep_sort_realtime) README.

## Tech stack

Python, PyTorch, torchvision, Ultralytics YOLOv8, deep-sort-realtime, OpenCV, Streamlit, Flask, pandas, Plotly

## Screenshots

![](Crowd%20Management%20using%20Yolov8/Dashboard%20UI.png)

![](Crowd%20Management%20using%20Yolov8/Security%20Dashboard%20UI.png)

![](Crowd%20Management%20using%20Yolov8/Analytics%20UI.png)

## Demo video

[![Watch the video](https://img.youtube.com/vi/AMzIEpD2tVQ/maxresdefault.jpg)](https://youtu.be/AMzIEpD2tVQ)
