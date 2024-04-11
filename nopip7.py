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

def detect_embryos(frame, min_area=100, max_area=10000):
    # Resize the frame to a standard width
    resized_frame = resize_frame(frame)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, adaptive_threshold_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the noise-reduced image
    contours, _ = cv2.findContours(adaptive_threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area
    filtered_contours = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]

    return filtered_contours, resized_frame

def detect_fluorescence(resized_frame, embryo_contours, intensity_threshold=cv2.THRESH_BINARY + cv2.THRESH_OTSU): #int threshold messed 
    int_threshold = intensity_threshold
    # Create a mask to focus on the embryo regions
    mask = np.zeros_like(resized_frame[:, :, 0], dtype=np.uint8)
    cv2.drawContours(mask, embryo_contours, -1, 255, -1)

    # Apply the mask to the resized frame
    masked_frame = cv2.bitwise_and(resized_frame, resized_frame, mask=mask)

    # Convert the masked frame to the HSV color space
    hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)

    # Define the color range for the fluorescent specks - this is in HSV color space
    lower_color = np.array([0, intensity_threshold, 0]) # messed with this
    upper_color = np.array([150, 200, 150]) # this too 

    # Threshold the HSV image to get only the fluorescent regions
    binary = cv2.inRange(hsv, lower_color, upper_color)

    # Find contours in the binary image
    fluorescent_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding rectangles around the fluorescent regions on the original resized frame
    for contour in fluorescent_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(resized_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    # Calculate the fluorescence level
    fluorescent_pixels = hsv[binary > 0]
    fluorescence_level = np.mean(fluorescent_pixels) if len(fluorescent_pixels) > 0 else 0

    return resized_frame, fluorescence_level, int_threshold

# Video source selection
video_source = st.radio("Select Video Source", ("Webcam", "Upload"))

if video_source == "Webcam":
    video_capture = cv2.VideoCapture(0)
else:
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
    if uploaded_file is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getvalue())
        video_capture = cv2.VideoCapture("temp_video.mp4")

if 'video_capture' in locals() and video_capture.isOpened():
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        embryo_contours, resized_frame = detect_embryos(frame)

        if len(embryo_contours) > 0:
            processed_frame, fluorescence_level, int_threshold = detect_fluorescence(resized_frame, embryo_contours)
            st.write("Fluorescence Level: {:.2f}".format(fluorescence_level))
            st.write("Int threshold:{:.2f}".format(int_threshold))
            st.image(processed_frame, channels="BGR", use_column_width=True)
        else:
            st.write("No embryos detected")

        time.sleep(1)  # Slow down the loop for demonstration purposes

    video_capture.release()
else:
    st.write("Failed to capture frame from video source.")
