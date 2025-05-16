# Python In-built packages
from pathlib import Path
import PIL
import cv2
import tempfile
import numpy as np

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Video Safety Detection using YOLOv8",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("🎥 Video Upload - Object Detection & Safety Score using YOLOv8")

# Sidebar
st.sidebar.header("⚙️ ML Model Config")

# Model Options
model_type = st.sidebar.radio("Select Task", ['Detection', 'Segmentation'])
confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
model_path = Path(settings.DETECTION_MODEL if model_type == 'Detection' else settings.SEGMENTATION_MODEL)

# Load Pre-trained Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("📤 Upload Video")
uploaded_video = st.sidebar.file_uploader("Choose a video file", type=("mp4", "avi", "mov", "mkv"))

if uploaded_video is not None:
    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    st.video(video_path)

    if st.sidebar.button("🔍 Detect Objects and Get Safety Score"):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        people_count = 0
        vehicle_count = 0
        streetlight_count = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame_count > 100:  # Limit to 100 frames to speed up
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(frame_rgb, conf=confidence)
            boxes = results[0].boxes
            class_ids = boxes.cls.tolist()

            # Count specific objects using class labels (COCO format assumed)
            for cls_id in class_ids:
                name = model.names[int(cls_id)]
                if name in ["person"]:
                    people_count += 1
                elif name in ["car", "bus", "truck", "motorbike", "bicycle"]:
                    vehicle_count += 1
                elif name in ["streetlight", "traffic light"]:  # Depending on the model class list
                    streetlight_count += 1

            frame_count += 1

        cap.release()

        # Display object counts
        st.subheader("📊 Object Counts Summary")
        st.write(f"👤 People Detected: **{people_count}**")
        st.write(f"🚗 Vehicles Detected: **{vehicle_count}**")
        st.write(f"💡 Streetlights Detected: **{streetlight_count}**")

        # Calculate Safety Score (Sample logic: you can customize)
        safety_score = (streetlight_count * 2 + people_count - vehicle_count) / 10
        safety_score = max(0, min(10, round(safety_score, 2)))  # Clamp between 0 and 10

        st.success(f"🛡️ Estimated Safety Score: **{safety_score} / 10**")

        # Optional interpretation
        if safety_score >= 8:
            st.markdown("✅ **Very Safe**")
        elif safety_score >= 5:
            st.markdown("⚠️ **Moderately Safe**")
        else:
            st.markdown("🚨 **Unsafe Area**")

else:
    st.info("Please upload a video to start detection.")
