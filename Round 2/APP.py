import os
import sys
import uuid

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'tumour.h5')
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
ALLOWED = {'.jpg', '.jpeg', '.png', '.bmp'}

# same size and class order as in tumour_detection.ipynb
IMG_SIZE = (150, 150)
CLASSES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
NAMES = {
    'glioma_tumor': 'Glioma tumour',
    'meningioma_tumor': 'Meningioma tumour',
    'no_tumor': 'No tumour',
    'pituitary_tumor': 'Pituitary tumour',
}

if not os.path.exists(MODEL_PATH):
    sys.exit('tumour.h5 not found. Train it with tumour_detection.ipynb and put it next to APP.py')

model = tf.keras.models.load_model(MODEL_PATH)

# model that also gives the last conv layer output, used for the heatmap
last_conv = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]
grad_model = tf.keras.models.Model(model.inputs, [last_conv.output, model.output])

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024


def load_image(path):
    # same preprocessing as the test generator: rgb, 150x150, scaled to 0-1
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    x = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(x, axis=0)


def grad_cam(x, class_idx):
    # Grad-CAM: weight the last conv maps by how much they push the class score
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    cam = tf.maximum(cam, 0) / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()


def save_overlay(path, cam, out_path):
    img = cv2.imread(path)
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heat, 0.4, 0)
    cv2.imwrite(out_path, overlay)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    file = request.files.get('image')
    if file is None or file.filename == '':
        return render_template('index.html', error='Please choose an image first.')

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED:
        return render_template('index.html', error='Only jpg, png and bmp images are supported.')

    name = uuid.uuid4().hex
    img_path = os.path.join(UPLOAD_DIR, name + ext)
    file.save(img_path)

    try:
        x = load_image(img_path)
    except Exception:
        os.remove(img_path)
        return render_template('index.html', error='Could not read that image.')

    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = CLASSES[idx]

    scores = sorted(
        [(NAMES[c], float(p) * 100) for c, p in zip(CLASSES, probs)],
        key=lambda s: s[1], reverse=True)

    heatmap = None
    if label != 'no_tumor':
        heatmap = name + '_cam.jpg'
        save_overlay(img_path, grad_cam(x, idx), os.path.join(UPLOAD_DIR, heatmap))

    return render_template(
        'index.html',
        image='uploads/' + name + ext,
        heatmap='uploads/' + heatmap if heatmap else None,
        label=NAMES[label],
        tumour=label != 'no_tumor',
        confidence=float(probs[idx]) * 100,
        scores=scores,
    )


if __name__ == '__main__':
    app.run(debug=True)
