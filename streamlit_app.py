import os
import cv2
import streamlit as st
import numpy as np
from io import BytesIO
from PIL import Image
from pdf2image import convert_from_path, convert_from_bytes
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.exceptions import HttpResponseError
# import matplotlib.pyplot as plt
# # from openai import OpenAI
# import base64
# import requests

# set up endpoint and key
endpoint = st.secrets['FORM_RECOGNIZER_ENDPOINT']
key = st.secrets['FORM_RECOGNIZER_KEY']

# define directories
ballots_folder = "ballots"
temp_folder = "temp"

document_analysis_client = DocumentAnalysisClient(
    endpoint=endpoint, credential=AzureKeyCredential(key))


def load_example_images():
    example_images = []
    for filename in os.listdir(ballots_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', 'pdf')):
            file_path = os.path.join(ballots_folder, filename)
            example_images.append(file_path)
    return example_images


def load_image(image_data):
    byte_images = []

    if isinstance(image_data, str):  # handling file path
        if image_data.lower().endswith('.pdf'):
            images = convert_from_path(image_data)
            for img in images:
                output = BytesIO()
                img.save(output, format="PNG")
                byte_images.append(output.getvalue())
        else:  # it's an image path
            with open(image_data, 'rb') as file:
                byte_images.append(file.read())

    else:  # handling UploadedFile object
        if image_data.type == "application/pdf":
            images = convert_from_bytes(image_data.getvalue())
            for img in images:
                output = BytesIO()
                img.save(output, format="PNG")
                byte_images.append(output.getvalue())
        else:  # it's an image
            byte_images.append(image_data.getvalue())

    return byte_images


def analyze_document(content):
    poller = document_analysis_client.begin_analyze_document(
        "prebuilt-layout", content)
    return poller.result()


def display_annotated_image(image_bytes, analyze_result):
    image = cv2.imdecode(np.frombuffer(
        image_bytes, np.uint8), cv2.IMREAD_COLOR)
    for page in analyze_result.pages:
        for word_info in page.words:
            pts = np.array(word_info.polygon, np.int32).reshape((-1, 1, 2))
            image = cv2.polylines(image, [pts], True, (0, 255, 0), 2)
        for selection_mark in page.selection_marks:
            selection_pts = np.array(selection_mark.polygon, np.int32).reshape(
                (-1, 1, 2))
            image = cv2.polylines(image, [selection_pts], True, (0, 0, 255), 2)

    return image



def analyze_document_opencv(image_bytes):
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Find contours
    max_area = 0
    c = 0
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in contours:
        area = cv2.contourArea(i)
        if area > 1000:
            if area > max_area:
                max_area = area
                best_cnt = i
                image = cv2.drawContours(image, contours, c, (0, 255, 0), 3)
        c += 1

    # Create a mask to search only within these boundaries
    mask = np.zeros((gray.shape), np.uint8)
    cv2.drawContours(mask, [best_cnt], 0, 255, -1)
    cv2.drawContours(mask, [best_cnt], 0, 0, 2)


    # Cut away the identified mask from the image
    out = np.zeros_like(gray)
    out[mask == 255] = gray[mask == 255]


    # Apply blur and adaptive threshold to this new image
    blur = cv2.GaussianBlur(out, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)


    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    c = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 3000/2:
            cv2.drawContours(image, contours, c, (0, 255, 0), 3)
        c += 1

    return image, contours


# def gpt_groupings(ballot_image, ballot_text):
#     openai_api_key = st.secrets['OPENAI_KEY']

#     assert isinstance(
#         ballot_image, bytes), "ballot_image must be of type bytes"

#     base64_image = base64.b64encode(ballot_image).decode('utf-8')

#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {openai_api_key}"
#     }

#     prompt_text = (
#         "Below is the text from a ballot paper. Each section of the ballot contains multiple lines."
#         "Please provide the line numbers that are associated with every section. Please make sure you give a mapping for every section in the document"
#         "Please return the data in the following fomat {'section name': 'lines'}. DO NOT PROVIDE ANY OTHER TEXT OTHER THAN THE JSON OBJECT"
#         "Here's the text from the ballot:\n\n" + ballot_text)

#     payload = {
#         "model":
#         "gpt-4-vision-preview",
#         "temperature": 0,
#         "messages": [{
#             "role":
#             "user",
#             "content": [{
#                 "type": "text",
#                 "text": prompt_text
#             }, {
#                 "type": "image_url",
#                 "image_url": {
#                     "url": f"data:image/jpeg;base64,{base64_image}"
#                 }
#             }]
#         }],
#         "max_tokens":
#         1000,
#     }

#     response = requests.post("https://api.openai.com/v1/chat/completions",
#                              headers=headers,
#                              json=payload)
#     print("this is the resonse!!!!!")
#     print(response)

#     return response.json()


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
                except HttpResponseError as err:
                    st.error(f"Failed to analyze page {idx+1} in document: {err.response.reason}")

            # Use OpenCV to find and draw contours
            opencv_image, contours = analyze_document_opencv(content)

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
        # for page in result.pages:
        #     for line in page.lines:
        #         #     st.write(line.content)
        #         lines += str(line_counter) + ": " + line.content + "\n"
        #         line_counter += 1

        # groupings = gpt_groupings(content, lines)
        # st.header("GPT Groupings")
        # st.write(groupings['choices'][0]['message']['content'])
        # st.write(lines)
        # for page in result.pages:
        #     for line in page.lines:
        #         st.write(line.content)

    st.write("----------------------------------------")
