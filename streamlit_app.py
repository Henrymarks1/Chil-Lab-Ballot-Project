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
from openai import OpenAI
import base64
import requests
from dotenv import load_dotenv
load_dotenv()


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


def gpt_groupings(ballot_image, ballot_text):
    openai_api_key = st.secrets['OPENAI_KEY']

    assert isinstance(
        ballot_image, bytes), "ballot_image must be of type bytes"

    base64_image = base64.b64encode(ballot_image).decode('utf-8')

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    prompt_text = (
        "Below is the text from a ballot paper. Each section of the ballot contains multiple lines."
        "Please provide the line numbers that are associated with every section. Please make sure you give a mapping for every section in the document"
        "Please return the data in the following fomat {'section name': 'lines'}. DO NOT PROVIDE ANY OTHER TEXT OTHER THAN THE JSON OBJECT"
        "Here's the text from the ballot:\n\n" + ballot_text)

    payload = {
        "model":
        "gpt-4-vision-preview",
        "temperature": 0,
        "messages": [{
            "role":
            "user",
            "content": [{
                "type": "text",
                "text": prompt_text
            }, {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }]
        }],
        "max_tokens":
        1000,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions",
                             headers=headers,
                             json=payload)
    print("this is the resonse!!!!!")
    print(response)

    return response.json()


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
                result = analyze_document(content)
            except HttpResponseError as err:
                st.error(
                    f"Failed to analyze page {idx+1} in document: {err.response.reason}"
                )

        # display the results
        with st.spinner(f"Preparing annotated image for page {idx+1}..."):
            annotated_image = display_annotated_image(content, result)
            annotated_image = Image.fromarray(annotated_image)
            img_buf = BytesIO()
            annotated_image.save(img_buf, format="PNG")
            st.image(img_buf,
                     caption=f"Annotated Image for page {idx+1}",
                     use_column_width=True)

        lines = ""
        line_counter = 1
        for page in result.pages:
            for line in page.lines:
                #     st.write(line.content)
                lines += str(line_counter) + ": " + line.content + "\n"
                line_counter += 1

        groupings = gpt_groupings(content, lines)
        st.header("GPT Groupings")
        st.write(groupings['choices'][0]['message']['content'])
        st.write(lines)
        # for page in result.pages:
        #     for line in page.lines:
        #         st.write(line.content)

    st.write("----------------------------------------")
