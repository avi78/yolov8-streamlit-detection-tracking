from ultralytics import YOLO
import streamlit as st
import cv2
import yt_dlp
import settings
import tempfile


def load_model(model_path):
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = display_tracker == 'Yes'
    tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml")) if is_display_tracker else None
    return is_display_tracker, tracker_type


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    image = cv2.resize(image, (720, int(720 * (9 / 16))))
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        res = model.predict(image, conf=conf)
    res_plotted = res[0].plot()
    st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_column_width=True)


def get_youtube_stream_url(youtube_url):
    ydl_opts = {'format': 'best[ext=mp4]', 'no_warnings': True, 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']


def play_youtube_video(conf, model):
    source_youtube = st.sidebar.text_input("YouTube Video url")
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        if not source_youtube:
            st.sidebar.error("Please enter a YouTube URL")
            return
        try:
            st.sidebar.info("Extracting video stream URL...")
            stream_url = get_youtube_stream_url(source_youtube)
            st.sidebar.info("Opening video stream...")
            vid_cap = cv2.VideoCapture(stream_url)
            if not vid_cap.isOpened():
                st.sidebar.error("Failed to open video stream. Try a different video.")
                return
            st.sidebar.success("Video stream opened successfully!")
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
                else:
                    break
            vid_cap.release()
        except Exception as e:
            st.sidebar.error(f"An error occurred: {str(e)}")


def play_rtsp_stream(conf, model):
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption('Example: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_webcam(conf, model):
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_stored_video(conf, model):
    source_vid = st.sidebar.selectbox("Choose a video...", settings.VIDEOS_DICT.keys())
    is_display_tracker, tracker = display_tracker_options()
    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)
    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_uploaded_video(conf, model, uploaded_file):
    """
    Detect objects in a user-uploaded video and calculate a safety score.
    """
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Uploaded Video'):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        vid_cap = cv2.VideoCapture(video_path)
        st_frame = st.empty()

        total_frames = 0
        person_count = 0
        vehicle_count = 0
        light_count = 0

        while vid_cap.isOpened():
            success, frame = vid_cap.read()
            if not success:
                break

            total_frames += 1

            if is_display_tracker:
                res = model.track(frame, conf=conf, persist=True, tracker=tracker)
            else:
                res = model.predict(frame, conf=conf)

            res_plotted = res[0].plot()
            st_frame.image(res_plotted, channels="BGR", use_column_width=True)

            for box in res[0].boxes:
                cls = int(box.cls[0].item())
                name = model.names[cls]
                if name.lower() == 'person':
                    person_count += 1
                elif name.lower() in ['car', 'truck', 'bus', 'motorbike']:
                    vehicle_count += 1
                elif name.lower() in ['street light', 'light']:
                    light_count += 1

        vid_cap.release()

        score = (person_count + light_count * 2 - vehicle_count) / max(total_frames, 1)
        safety_score = round(max(min(score * 10, 10), 0), 2)

        st.success("✅ Detection Completed!")
        st.info(f"👤 Persons Detected: {person_count}")
        st.info(f"🚗 Vehicles Detected: {vehicle_count}")
        st.info(f"💡 Lights Detected: {light_count}")
        st.warning(f"🛡️ Estimated Safety Score (0-10): {safety_score}")
