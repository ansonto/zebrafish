# report1 
# from nopip3 - ie. good edge detection 
import cv2
import numpy as np
import streamlit as st

st.title("Zebrafish Embryo Fluorescence Detector")

def is_fluorescence_detected(frame, threshold=1500):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_frame, 50, 255, cv2.THRESH_BINARY)
    num_pixels = cv2.countNonZero(binary)
    return num_pixels > threshold

def process_frame(frame, min_fluorescence=100, max_fluorescence=255):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, adaptive_threshold_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the noise-reduced image
    contours, _ = cv2.findContours(adaptive_threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contours of the embryos
    embryo_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Adjust this value based on the size of the embryos
            embryo_contours.append(contour)

    # Draw contours and bounding rectangles on the original image
    for contour in embryo_contours:
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Detect fluorescence
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, min_fluorescence, max_fluorescence, cv2.THRESH_BINARY)
    fluorescent_pixels = binary[binary > min_fluorescence]
    fluorescence_level = np.mean(fluorescent_pixels)

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

if 'video_capture' in locals():
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        if is_fluorescence_detected(frame):
            processed_frame, fluorescence_level = process_frame(frame)
            st.write("Fluorescence Level: {:.2f}".format(fluorescence_level))
            st.image(processed_frame, channels="BGR", use_column_width=True)
        else:
            st.write("No fluorescence detected")

    # Release the video capture
    video_capture.release()
else:
    st.write("Failed to capture frame from video source.")