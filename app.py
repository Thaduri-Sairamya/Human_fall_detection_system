import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from io import BytesIO
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="Human Fall Detection",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Human Fall Detection System")
st.write("AI-powered human fall detection using YOLO and Computer Vision.")
st.caption("Upload an image or video to run detection with your trained model.")
st.divider()

MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Copy your trained YOLO weights into this folder "
        "and name the file best.pt."
    )
    st.stop()

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

def is_fall_class(class_name):
    """Return True when a model class label represents a fall."""
    name = str(class_name).lower().replace("_", " ").replace("-", " ").strip()
    negative_labels = ("not fall", "no fall", "non fall", "normal", "standing")
    if any(label in name for label in negative_labels):
        return False
    return any(word in name for word in ("fall", "falling", "fallen"))

st.sidebar.header("Detection settings")
confidence = st.sidebar.slider(
    "Confidence threshold", 0.10, 1.00, 0.50, 0.05
)
st.sidebar.info(
    "The model must have a trained fall-related class. A person-only model "
    "does not determine whether someone has fallen."
)

mode = st.radio(
    "Choose detection mode",
    ["Image detection", "Video detection"],
    horizontal=True
)

if mode == "Image detection":
    uploaded_image = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        left, right = st.columns(2)
        with left:
            st.subheader("Original image")
            st.image(image, use_container_width=True)

        if st.button("Detect fall", type="primary"):
            results = model.predict(
                source=np.array(image), conf=confidence, verbose=False
            )
            result = results[0]
            annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

            fall_count = 0
            other_count = 0
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    if is_fall_class(class_name):
                        fall_count += 1
                    else:
                        other_count += 1

            with right:
                st.subheader("Detection result")
                st.image(annotated, use_container_width=True)

            if fall_count:
                st.error(f"🚨 Fall-class detections: {fall_count}")
            else:
                st.success("No fall-class detection found.")

            c1, c2 = st.columns(2)
            c1.metric("Fall detections", fall_count)
            c2.metric("Other detections", other_count)

            output = BytesIO()
            Image.fromarray(annotated).save(output, format="JPEG")
            st.download_button(
                "Download annotated image",
                data=output.getvalue(),
                file_name="fall_detection_result.jpg",
                mime="image/jpeg"
            )

else:
    uploaded_video = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("Start video detection", type="primary"):
            input_path = None
            output_path = None
            cap = None
            writer = None

            try:
                suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as temp_input:
                    temp_input.write(uploaded_video.getvalue())
                    input_path = temp_input.name

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as temp_output:
                    output_path = temp_output.name

                cap = cv2.VideoCapture(input_path)
                if not cap.isOpened():
                    st.error("Unable to open the uploaded video.")
                    st.stop()

                fps = cap.get(cv2.CAP_PROP_FPS)
                if not fps or fps <= 0:
                    fps = 25.0

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_count = 0
                fall_frames = 0

                progress = st.progress(0)
                status = st.empty()

                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    result = model.predict(
                        source=frame, conf=confidence, verbose=False
                    )[0]
                    annotated = result.plot()

                    if writer is None:
                        height, width = annotated.shape[:2]
                        writer = cv2.VideoWriter(
                            output_path,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (width, height)
                        )

                    frame_has_fall = False
                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            if is_fall_class(result.names[class_id]):
                                frame_has_fall = True
                                break

                    if frame_has_fall:
                        fall_frames += 1

                    writer.write(annotated)
                    frame_count += 1

                    if frame_count % 5 == 0:
                        progress.progress(
                            min(frame_count / max(total_frames, 1), 1.0)
                        )
                        status.write(
                            f"Processed {frame_count} / {total_frames or '?'} frames"
                        )

                if cap is not None:
                    cap.release()
                    cap = None
                if writer is not None:
                    writer.release()
                    writer = None

                progress.progress(1.0)

                if frame_count == 0 or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                    st.error("No output video was generated. Check the input video and codec support.")
                else:
                    with open(output_path, "rb") as file:
                        video_bytes = file.read()

                    status.success("Video processing completed.")
                    st.subheader("Processed video")
                    st.video(video_bytes)

                    c1, c2 = st.columns(2)
                    c1.metric("Frames processed", frame_count)
                    c2.metric("Frames with fall detection", fall_frames)

                    if fall_frames:
                        st.error("🚨 Fall-class detection appeared in one or more frames.")
                    else:
                        st.success("No fall-class detection found.")

                    st.download_button(
                        "Download processed video",
                        data=video_bytes,
                        file_name="fall_detection_output.mp4",
                        mime="video/mp4"
                    )

            except Exception as exc:
                st.error(f"Video processing failed: {exc}")
            finally:
                if cap is not None:
                    cap.release()
                if writer is not None:
                    writer.release()
                for path in (input_path, output_path):
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except OSError:
                            pass

st.divider()
st.caption(
    "Human Fall Detection | YOLO | OpenCV | Streamlit. "
    "This demo is not a substitute for a monitored safety system."
)
