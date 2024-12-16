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

#
# def process_ballot_page(content, idx, display_name):
#     """Process a single ballot page with proper error handling"""
#     try:
#         # Convert bytes to image for OpenCV processing
#         nparr = np.frombuffer(content, np.uint8)
#         image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#
#         if image is None:
#             raise ValueError(f"Failed to decode image for page {idx + 1}")
#
#         # Detect features and get masked image
#         feature_regions, masked_image = feature_detector.detect_features(image)
#
#         # Convert masked image back to bytes
#         success, buffer = cv2.imencode('.png', masked_image)
#         if not success:
#             raise ValueError(f"Failed to encode masked image for page {idx + 1}")
#
#         # Process with Azure Form Recognizer
#         azure_result = analyze_document(content)  # Using updated analyze_document with compression
#         page_word_locations = extract_words_and_coordinates(azure_result)
#
#         # Display results
#         debug_image = image.copy()
#         for x, y, w, h in feature_regions['ballot_images']:
#             cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
#
#         st.image(Image.fromarray(debug_image), caption="Detected Features", use_column_width=True)
#         st.image(Image.fromarray(masked_image), caption="Masked Image", use_column_width=True)
#
#         result_image = display_annotated_image(buffer.tobytes(), azure_result)
#         st.image(Image.fromarray(result_image), caption="Processed Result", use_column_width=True)
#
#         return page_word_locations
#
#     except Exception as e:
#         st.error(f"Error processing page {idx + 1}: {str(e)}")
#         return []
#

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

# import os
# import cv2
# import streamlit as st
# import numpy as np
# from io import BytesIO
# from PIL import Image
# from azure.core.exceptions import HttpResponseError
#
# ## Function imports
# from utility.azure_utility import analyze_document, display_annotated_image
# from utility.word_extraction_utlity import create_downloadable_dataframe, extract_words_and_coordinates
# from utility.image_utility import load_example_images, load_image
# from utility.open_cv_utility import analyze_document_opencv
#
# st.title("Ballot Paper Parsing")
#
# # Load examples
# example_images = load_example_images()
#
# # Dropdown to select an example
# selected_example = st.selectbox("Choose from the example ballot papers",
#                                 options=example_images,
#                                 format_func=lambda x: os.path.basename(x))
#
# # Upload a new document
# uploaded_file = st.file_uploader("Or upload a new document",
#                                  type=["pdf", "jpg", "png"])
#
# if st.button('Start Analysis'):
#     word_locations = []
#     all_boxes = []
#     all_horriz_lines = []
#     all_vert_lines = []
#
#     if uploaded_file or selected_example:
#         image_contents = load_image(uploaded_file if uploaded_file else selected_example)
#
#         if uploaded_file:
#             display_name = uploaded_file.name
#         else:
#             display_name = os.path.basename(selected_example)
#
#         for idx, content in enumerate(image_contents):
#             with st.spinner(f"Analyzing page {idx+1} in {display_name}..."):
#                 try:
#                     azure_result = analyze_document(content)
#                     word_locations = extract_words_and_coordinates(azure_result)
#                 except HttpResponseError as err:
#                     st.error(f"Failed to analyze page {idx+1} in document: {err.response.reason}")
#
#                 # Use OpenCV to find and draw contours
#                 opencv_image, boxes, horriz_lines, vertical_lines = analyze_document_opencv(content)
#                 all_horriz_lines.extend(horriz_lines)
#                 all_vert_lines.extend(vertical_lines)
#                 all_boxes.extend(boxes)
#
#                 # Display Azure's annotated image
#                 with st.spinner(f"Preparing annotated image for page {idx+1}..."):
#                     annotated_image = display_annotated_image(content, azure_result)
#                     # Combine annotated_image with opencv_image
#                     annotated_image_np = np.array(annotated_image)
#                     combined_image = cv2.addWeighted(annotated_image_np, 0.5, opencv_image, 0.5, 0)
#                     # Convert back to PIL Image to display in Streamlit
#                     combined_image_pil = Image.fromarray(combined_image)
#                     img_buf = BytesIO()
#                     combined_image_pil.save(img_buf, format="PNG")
#                     st.image(img_buf, caption=f"Combined Annotated Image for page {idx+1}", use_column_width=True)
#
#         csv_data = create_downloadable_dataframe(word_locations, all_boxes, all_horriz_lines, all_vert_lines)
#
#         st.download_button(
#             label="Download Data as CSV",
#             data=csv_data,
#             file_name="processed_data.csv",
#             mime='text/csv'
#         )
"""
NEW CHANGES TO find and handle extranous features
from utility.ballot_feature_detection import BallotFeatureDetector

# Initialize the detector
feature_detector = BallotFeatureDetector()

# In analysis loop:
for idx, content in enumerate(image_contents):
    with st.spinner(f"Analyzing page {idx+1} in {display_name}..."):
        # Convert bytes to image
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Detect extraneous features
        feature_regions = feature_detector.detect_features(image)

        # Get Azure results and OpenCV detections as before
        azure_result = analyze_document(content)
        word_locations = extract_words_and_coordinates(azure_result)
        opencv_image, boxes, horriz_lines, vertical_lines = analyze_document_opencv(content)

        # Filter out detections within feature regions
        filtered_word_locations = feature_detector.filter_detections(word_locations, feature_regions)
        filtered_boxes = feature_detector.filter_detections(boxes, feature_regions)
        filtered_horriz_lines = feature_detector.filter_detections(horriz_lines, feature_regions)
        filtered_vert_lines = feature_detector.filter_detections(vertical_lines, feature_regions)



"""
