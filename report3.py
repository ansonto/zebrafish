import cv2
import numpy as np
import streamlit as st
import time

st.title("Zebrafish Embryo Fluorescence Detector")

def resize_frame(frame, target_width=640):
    height, width = frame.shape[:2]
    ratio = target_width / width
    target_height = int(height * ratio)
    resized_frame = cv2.resize(frame, (target_width, target_height))
    return resized_frame

def detect_embryos(frame, background_subtractor):
    # Resize the frame to a standard width
    resized_frame = resize_frame(frame)

    # Apply background subtraction
    foreground_mask = background_subtractor.apply(resized_frame)

    # Apply morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)

    # Find contours of the foreground objects
    contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, resized_frame

def detect_fluorescence(frame, embryo_contours, intensity_threshold=225):
    # Create a mask to focus on the embryo regions
    mask = np.zeros_like(frame[:, :, 0], dtype=np.uint8)
    cv2.drawContours(mask, embryo_contours, -1, 255, -1)

    # Apply the mask to the frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

    # Convert the masked frame to the HSV color space
    hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)

    # Define the color range for the fluorescent specks
    lower_color = np.array([0, 0, intensity_threshold])
    upper_color = np.array([255, 255, 255])

    # Threshold the HSV image to identify the fluorescent regions
    binary = cv2.inRange(hsv, lower_color, upper_color)

    # Find contours of the fluorescent regions
    fluorescent_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding rectangles around the fluorescent regions on the original frame
    for contour in fluorescent_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Calculate the fluorescence level as the average intensity of the fluorescent regions
    fluorescent_pixels = masked_frame[binary > 0]
    fluorescence_level = np.mean(fluorescent_pixels) if len(fluorescent_pixels) > 0 else 0

    return frame, fluorescence_level

# Choose video source and capture a frame
video_source = st.radio("Select Video Source", ("Webcam", "Upload"))

if video_source == "Webcam":
    video_capture = cv2.VideoCapture(0)
else:
    # Video upload
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
    if uploaded_file is not None:
        # Save the uploaded file to a temporary directory
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getvalue())
        # Read video from uploaded file
        video_capture = cv2.VideoCapture("temp_video.mp4")

# Create a background subtractor
background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

if 'video_capture' in locals() and video_capture.isOpened():
    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_count % 30 == 0:  # Process one frame per second (assuming 30 FPS)
            embryo_contours, resized_frame = detect_embryos(frame, background_subtractor)

            if len(embryo_contours) > 0:
                processed_frame, fluorescence_level = detect_fluorescence(resized_frame, embryo_contours)
                st.write("Fluorescence Level: {:.2f}".format(fluorescence_level))
                st.image(processed_frame, channels="BGR", use_column_width=True)
            else:
                st.write("No embryos detected")

        frame_count += 1

    # Release the video capture
    video_capture.release()
else:
    st.write("Failed to capture frame from video source.")