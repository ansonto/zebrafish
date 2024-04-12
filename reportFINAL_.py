import cv2
import numpy as np
import streamlit as st
import time

st.title("Zebrafish Embryo Fluorescence Detector")

def is_embryo_present(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_frame, 50, 255, cv2.THRESH_BINARY)
    num_pixels = cv2.countNonZero(binary)
    return num_pixels > 1500

def process_frame(frame, min_fluorescence=200, max_fluorescence=255):
    #convert image to grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #noise reduction
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, adaptive_threshold_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #find contours in noise-reduced image
    contours, _ = cv2.findContours(adaptive_threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #find contours of embryos
    embryo_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  #adjust this value based on the size of the embryos
            embryo_contours.append(contour)

    if len(embryo_contours) > 0:
        #draw contours and bounding rectangles on the original image
        for contour in embryo_contours:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            #extract the contour region from the frame
            contour_region = frame[y:y+h, x:x+w]

            #convert the contour region to HSV color space
            hsv = cv2.cvtColor(contour_region, cv2.COLOR_BGR2HSV)

            #define color range for fluorescent specks
            lower_color = np.array([0, 0, min_fluorescence])
            upper_color = np.array([255, 255, max_fluorescence])

            #threshold the HSV image to identify fluorescent regions
            binary = cv2.inRange(hsv, lower_color, upper_color)

            #calculate fluorescence level as the average intensity of the fluorescent pixels
            fluorescent_pixels = contour_region[binary > 0]
            if len(fluorescent_pixels) > 0:
                fluorescence_level = np.mean(fluorescent_pixels)
                #draw the fluorescence level on the frame
                cv2.putText(frame, f"FL: {fluorescence_level:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No Fluorescence", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return frame
    else:
        return frame

#choose video source and capture a frame
video_source = st.radio("Select Video Source", ("Webcam", "Upload"))

if video_source == "Webcam":
    video_capture = cv2.VideoCapture(0)
else:
    #video upload
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
    if uploaded_file is not None:
        # Save the uploaded file to a temporary directory
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getvalue())
        #read video from uploaded file
        video_capture = cv2.VideoCapture("temp_video.mp4")

if 'video_capture' in locals() and video_capture.isOpened():
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)  #process one frame per second
    frame_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            if is_embryo_present(frame):
                st.write("Embryo Present")
                processed_frame = process_frame(frame)
                st.image(processed_frame, channels="BGR", use_column_width=True)
            else:
                st.write("No Embryo Detected")

        frame_count += 1

    #release the video capture
    video_capture.release()
else:
    st.write("Failed to capture frame from video source.")
