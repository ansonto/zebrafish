import cv2
import numpy as np
import streamlit as st
import time

st.title("Zebrafish Embryo Fluorescence Detector")

def detect_embryos(frame, min_area=100, max_area=10000):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, adaptive_threshold_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the noise-reduced image
    contours, _ = cv2.findContours(adaptive_threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area
    filtered_contours = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]

    return filtered_contours

def detect_fluorescence(frame, embryo_contours, intensity_threshold=225):
    # Create a mask to focus on the embryo regions
    mask = np.zeros_like(frame[:, :, 0])
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

if 'video_capture' in locals() and video_capture.isOpened():
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        embryo_contours = detect_embryos(frame)

        if len(embryo_contours) > 0:
            processed_frame, fluorescence_level = detect_fluorescence(frame, embryo_contours)
            st.write("Fluorescence Level: {:.2f}".format(fluorescence_level))
            st.image(processed_frame, channels="BGR", use_column_width=True)
        else:
            st.write("No embryos detected")

        # Add a delay of 1 second to achieve 1 FPS
        time.sleep(1)

    # Release the video capture
    video_capture.release()
else:
    st.write("Failed to capture frame from video source.")