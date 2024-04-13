import cv2
import numpy as np
import streamlit as st

st.title("Contour Extraction for Testing Code")

def extract_contours(gray_image, contour_area_threshold=100):
    #noise reduction
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, adaptive_threshold_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #find contours in the noise-reduced image
    contours, _ = cv2.findContours(adaptive_threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #find the contours of the objects
    object_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > contour_area_threshold:
            object_contours.append(contour)

    return object_contours

def visualize_contours(image, contours):
    #create a black image with the same dimensions as the input image
    contour_visualization = np.zeros_like(image)

    #draw the contours on the black image with white color
    cv2.drawContours(contour_visualization, contours, -1, (255, 255, 255), -1)

    return contour_visualization

#upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file is not None:
    #read uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    #convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #extract contours from the grayscale image
    contours = extract_contours(gray_image)

    #visualize contours
    contour_visualization = visualize_contours(image, contours)

    #display the original image and the contour visualization
    st.image(image, caption="Original Image", use_column_width=True)
    st.image(contour_visualization, caption="Contour Visualization", use_column_width=True)
else:
    st.warning("Please upload an image to get started.")