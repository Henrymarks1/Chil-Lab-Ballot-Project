import os
import cv2
import streamlit as st
import numpy as np
import gspread
from io import BytesIO
from PIL import Image

##function imports
from utility.azure_utility import analyze_document, display_annotated_image
from utility.google_sheets_utlity import extract_words_and_coordinates, write_excel
from utility.image_utility import load_example_images, load_image
from utility.open_cv_utility import analyze_document_opencv




st.title("Ballot Paper Parsing")

# Load examples
example_images = load_example_images()

# Dropdown to select an example
selected_example = st.selectbox("Choose from the example ballot papers",
                                options=example_images,
                                format_func=lambda x: os.path.basename(x))

# Upload a new document
uploaded_file = st.file_uploader("Or upload a new document",
                                 type=["pdf", "jpg", "png"])
# The google sheet
google_sheet_url = "https://docs.google.com/spreadsheets/d/18ebESsCRhphoTDL8PPboZM70nvXFL-wo-fMg0OA_t-k/edit?usp=sharing"
st.markdown(f"<a href='{google_sheet_url}' target='_blank'><button style='color: black; background-color: #0F9D58; border: none; padding: 10px 20px; text-align: center; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 12px;'>Open Google Sheet</button></a>", unsafe_allow_html=True)

if uploaded_file or selected_example:
    image_contents = load_image(
        uploaded_file if uploaded_file else selected_example)

    if uploaded_file:
        display_name = uploaded_file.name
    else:
        display_name = os.path.basename(selected_example)

        for idx, content in enumerate(image_contents):
            with st.spinner(f"Analyzing page {idx+1} in {display_name}..."):
                try:
                    azure_result = analyze_document(content)
                    extract_words_and_coordinates(azure_result)
                except HttpResponseError as err:
                    st.error(f"Failed to analyze page {idx+1} in document: {err.response.reason}")

            # Use OpenCV to find and draw contours
            opencv_image, boxes, lines = analyze_document_opencv(content)
        
            # Display Azure's annotated image
            with st.spinner(f"Preparing annotated image for page {idx+1}..."):
                annotated_image = display_annotated_image(content, azure_result)
                
                # Convert annotated_image to a format that can be combined with opencv_image
                annotated_image_np = np.array(annotated_image)
                combined_image = cv2.addWeighted(annotated_image_np, 0.5, opencv_image, 0.5, 0)

                # Convert back to PIL Image to display in Streamlit
                combined_image_pil = Image.fromarray(combined_image)
                img_buf = BytesIO()
                combined_image_pil.save(img_buf, format="PNG")
                st.image(img_buf, caption=f"Combined Annotated Image for page {idx+1}", use_column_width=True)

                lines = ""
                line_counter = 1
    
    #Write the data to an excel sheet
    write_excel()


