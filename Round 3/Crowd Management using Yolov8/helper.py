from ultralytics import YOLO
import streamlit as st
import cv2
from collections import defaultdict
import numpy as np
import csv
from datetime import datetime
from pathlib import Path
import settings


def check_camera_indices(cam_threshold):
    # try indices 0 to cam_threshold - 1 and keep the ones that open
    cam = []
    for i in range(int(cam_threshold)):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cam.append(i)
        cap.release()
    return cam


def load_model(model_path):
    # ultralytics downloads yolov8n.pt / yolov8n-seg.pt if the file is missing
    model = YOLO(model_path)
    return model


def display_tracker_options(key=""):
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'), key=f"tracker_radio{key}")
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"), key=f"tracker_type{key}")
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, track_history,
                             is_display_tracking=None, tracker=None, caption='Detected Video'):
    # Runs YOLOv8 on one frame, shows it and returns the number of persons in it.
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker, verbose=False)
    else:
        res = model.predict(image, conf=conf, verbose=False)

    boxes = res[0].boxes
    classes = boxes.cls.cpu().numpy().astype(int)
    count = sum(1 for c in classes if res[0].names[c] == "person")

    res_plotted = res[0].plot()

    # draw the last 30 centre points of every track
    if is_display_tracking and boxes.id is not None:
        track_ids = boxes.id.int().cpu().tolist()
        for box, track_id in zip(boxes.xywh.cpu(), track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track) > 30:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(res_plotted, [points], isClosed=False, color=(230, 230, 230), thickness=10)

    st_frame.image(res_plotted,
                   caption=caption,
                   channels="BGR",
                   use_column_width=True
                   )
    return count


# Define a function to write count and timestamp to a CSV file
def write_to_csv(count, csv_file):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_file = not Path(csv_file).exists()
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(["Timestamp", "Count"])
        writer.writerow([timestamp, count])


def show_crowd_status(t, w, count, crowd_threshold):
    t.markdown(f"Number of Persons: {count}")
    if count > crowd_threshold:
        w.warning(f"🚨 Alert: Crowd size exceeds the set threshold ({int(crowd_threshold)}) members 🚨")
    else:
        w.success("✅ Crowd is under control! ✅")


def play_webcam(confidence, model_path, cam_threshold, crowd_threshold, csv_file):
    # Runs detection on all selected webcams together. The total person count
    # is the sum over the cameras and it is logged to csv_file every frame.
    cam_list = check_camera_indices(cam_threshold)
    if not cam_list:
        st.sidebar.error("No webcam found")
        return

    selected = st.sidebar.multiselect("Cameras", cam_list, default=cam_list)
    is_display_tracker, tracker = display_tracker_options("_webcam")

    if st.sidebar.button('Start Cameras'):
        t = st.empty()
        w = st.empty()
        cols = st.columns(min(len(selected), 2) or 1)
        cams = []
        for n, index in enumerate(selected):
            cams.append({
                'index': index,
                'cap': cv2.VideoCapture(index),
                # separate model per camera so the tracker IDs stay apart
                'model': load_model(model_path),
                'frame': cols[n % len(cols)].empty(),
                'history': defaultdict(lambda: []),
            })
        try:
            while cams:
                total = 0
                for cam in list(cams):
                    success, image = cam['cap'].read()
                    if not success:
                        cam['cap'].release()
                        cams.remove(cam)
                        continue
                    total += _display_detected_frames(confidence, cam['model'], cam['frame'], image,
                                                      cam['history'], is_display_tracker, tracker,
                                                      caption=f"Camera {cam['index']}")
                show_crowd_status(t, w, total, crowd_threshold)
                write_to_csv(total, csv_file)
        except Exception as e:
            st.sidebar.error("Error loading webcam: " + str(e))
        finally:
            for cam in cams:
                cam['cap'].release()


def play_stored_video(conf, model, crowd_threshold, csv_file):
    # Plays one of the videos from settings.VIDEOS_DICT with detection/tracking.
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            t = st.empty()
            w = st.empty()
            track_history = defaultdict(lambda: [])
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    count = _display_detected_frames(conf,
                                                     model,
                                                     st_frame,
                                                     image,
                                                     track_history,
                                                     is_display_tracker,
                                                     tracker
                                                     )
                    show_crowd_status(t, w, count, crowd_threshold)
                    write_to_csv(count, csv_file)
                else:
                    vid_cap.release()
                    break

        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
