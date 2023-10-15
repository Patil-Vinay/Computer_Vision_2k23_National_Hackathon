# Python In-built packages
from pathlib import Path
from PIL import Image

import webbrowser  # Import the webbrowser module
# External packages
import streamlit as st

# Local Modules
import settings
import helper


# Setting page layout
st.set_page_config(
    page_title="Security Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def open_home_page():
    webbrowser.open("http://127.0.0.1:5000")
# Add a button to open a link to http://127.0.0.1:5000
if st.button("Home"):
    open_home_page()

# Main page heading
st.title("Security Dashboard")
st.sidebar.image("logo_2.png", use_column_width=True)
# Sidebar
st.sidebar.header("Parameter Tuning")
st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)
# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection', 'Segmentation'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

crowd_threshold = float(st.sidebar.slider(
    "Select Maximum Crowd Limit", 1, 500, 50))

cam_threshold = float(st.sidebar.slider(
    "Select Camera Count", 1, 50, 10))

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)
    st.stop()

csv_file = settings.CSV_FILE

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                with st.expander("Detection Results"):
                    for box in boxes:
                        st.write(box.data)

                # only count the person class
                count = sum(1 for c in boxes.cls.tolist() if res[0].names[int(c)] == "person")

                with st.expander("Number of people"):
                    st.write(count)

                if count > crowd_threshold:
                    st.warning(f"🚨 Alert: Crowd size exceeds the set threshold ({int(crowd_threshold)}) members 🚨")
                else:
                    st.success("✅ Crowd is under control! ✅")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model, crowd_threshold, csv_file)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model_path, cam_threshold, crowd_threshold, csv_file)

else:
    st.error("Please select a valid source type!")
