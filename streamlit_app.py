import os
import cv2
import streamlit as st
import numpy as np
from io import BytesIO
from PIL import Image
from azure.core.exceptions import HttpResponseError

from utility.azure_utility import (
    analyze_document,
    display_annotated_image,
    BallotFeatureDetector
)
from utility.image_processing import ImageProcessor
from utility.word_extraction_utlity import create_downloadable_dataframe, extract_words_and_coordinates
from utility.image_utility import load_example_images, load_image
from utility.open_cv_utility import analyze_document_opencv


def process_ballot_page(content, idx, display_name):
    try:
        # Process with Azure Form Recognizer first
        azure_result = analyze_document(content)

        # Convert bytes to image for OpenCV processing
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Failed to decode image for page {idx + 1}")

        # Pass both image and azure_result to detect_features
        feature_regions, masked_image = feature_detector.detect_features(image, azure_result)

        # Convert masked image back to bytes
        success, buffer = cv2.imencode('.png', masked_image)
        if not success:
            raise ValueError(f"Failed to encode masked image for page {idx + 1}")

        page_word_locations = extract_words_and_coordinates(azure_result)

        # Display results
        debug_image = image.copy()
        for x, y, w, h in feature_regions['ballot_images']:
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        st.image(Image.fromarray(debug_image), caption="Detected Features", use_column_width=True)
        st.image(Image.fromarray(masked_image), caption="Masked Image", use_column_width=True)

        result_image = display_annotated_image(buffer.tobytes(), azure_result)
        st.image(Image.fromarray(result_image), caption="Processed Result", use_column_width=True)

        return page_word_locations

    except Exception as e:
        st.error(f"Error processing page {idx + 1}: {str(e)}")
        return []

def analyze_ballot(content, display_name, idx):
    """Analyze a single ballot page with proper image processing."""
    image_processor = ImageProcessor()
    feature_detector = BallotFeatureDetector()

    try:
        # First resize/optimize the image if needed
        processed_content = image_processor.resize_if_needed(content)

        # Convert bytes to image for OpenCV processing
        nparr = np.frombuffer(processed_content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Failed to decode image for page {idx + 1}")

        # Detect features and get masked image
        feature_regions, masked_image = feature_detector.detect_features(image)

        # Convert masked image back to bytes for Azure analysis
        success, buffer = cv2.imencode('.png', masked_image)
        if not success:
            raise ValueError(f"Failed to encode masked image for page {idx + 1}")

        cleaned_content = buffer.tobytes()

        # Analyze with Azure
        azure_result = analyze_document(cleaned_content)
        page_word_locations = extract_words_and_coordinates(azure_result)

        # Display original image with detected features
        debug_image = image.copy()
        for x, y, w, h in feature_regions['ballot_images']:
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        debug_image_pil = Image.fromarray(debug_image)
        st.image(debug_image_pil, caption="Detected Features", use_column_width=True)

        # Show masked result
        masked_image_pil = Image.fromarray(masked_image)
        st.image(masked_image_pil, caption="Masked Image", use_column_width=True)

        # Show processed result
        result_image = display_annotated_image(cleaned_content, azure_result)
        result_image_pil = Image.fromarray(result_image)
        st.image(result_image_pil, caption="Processed Result", use_column_width=True)

        return page_word_locations

    except Exception as e:
        st.error(f"Error processing page {idx + 1}: {str(e)}")
        return []


st.title("Ballot Paper Parsing")

# Initialize the feature detector
feature_detector = BallotFeatureDetector()

# Load examples
example_images = load_example_images()

# Dropdown to select an example
selected_example = st.selectbox(
    "Choose from the example ballot papers",
    options=example_images,
    format_func=lambda x: os.path.basename(x)
)

# Upload a new document
uploaded_file = st.file_uploader(
    "Or upload a new document",
    type=["pdf", "jpg", "png"]
)

# Update the main analysis button logic in streamlit_app.py
if st.button('Start Analysis'):
    word_locations = []
    all_boxes = []
    all_horriz_lines = []
    all_vert_lines = []

    if uploaded_file or selected_example:
        image_contents = load_image(uploaded_file if uploaded_file else selected_example)
        display_name = uploaded_file.name if uploaded_file else os.path.basename(selected_example)

        for idx, content in enumerate(image_contents):
            with st.spinner(f"Analyzing page {idx + 1} in {display_name}..."):
                page_word_locations = process_ballot_page(content, idx, display_name)
                word_locations.extend(page_word_locations)

        # Create downloadable data
        csv_data = create_downloadable_dataframe(
            word_locations, all_boxes, all_horriz_lines, all_vert_lines)

        st.download_button(
            label="Download Data as CSV",
            data=csv_data,
            file_name="processed_data.csv",
            mime='text/csv'
        )

