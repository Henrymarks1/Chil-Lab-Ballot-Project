import cv2
import numpy as np
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import streamlit as st


# set up endpoint and key

endpoint = st.secrets['FORM_RECOGNIZER_ENDPOINT']
key = st.secrets['FORM_RECOGNIZER_KEY']

# define directories
ballots_folder = "ballots"
temp_folder = "temp"

document_analysis_client = DocumentAnalysisClient(
    endpoint=endpoint, credential=AzureKeyCredential(key))




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


