# Computer Vision 2K23 National Hackathon

Code from our team, Pixel Pioneers, for the Computer Vision 2K23 National Hackathon. 305 teams from across India took part and the hackathon had three rounds.

Team: Vinay Patil, Aditya Rane, Prachi Lal (Symbiosis Institute of Technology, Pune)

| Round | Task | Our rank |
| --- | --- | --- |
| 1 | Quiz on computer vision fundamentals (multiple correct MCQs) | 23 |
| 2 | Detection and classification of tumours in medical images | 6 |
| 3 | Person tracking with multiple overhead cameras | 4 |

## Round 1 - Quiz

Only a quiz, so there is no code for it.

## [Round 2](Round%202) - Brain tumour detection

A CNN trained in TensorFlow/Keras that classifies brain MRI scans as glioma, meningioma, pituitary tumour or no tumour. A Flask web app takes an uploaded scan, shows the predicted class with the score for each class and draws a Grad-CAM heatmap of the region the model used.

See [Round 2/readme.md](Round%202/readme.md).

## [Round 3](Round%203) - Person tracking with overhead cameras

- A crowd management web app with YOLOv8: Streamlit security dashboard for detection, tracking, person counting and crowd limit alerts on images, videos or several webcams, plus an analytics dashboard that plots the logged counts.
- Two camera stitching into one panorama with motion detection.
- Two camera person tracking with a torchvision detector and DeepSORT re-identification.

See [Round 3/readme.md](Round%203/readme.md).

## Repo layout

```
Round 2/
    tumour_detection.ipynb   training notebook
    APP.py                   Flask app
    templates/, static/
Round 3/
    Crowd Management using Yolov8/
        app.py               Flask home page
        streamlit_app.py     security dashboard
        crowd_analytics.py   analytics dashboard
        realtime_stitching.py
    REID/
        deep.py              two camera DeepSORT tracking
```

Each round has its own `requirements.txt`.
