"""THIS CODE DOESN'T WORK"""
# import fitz
# import cv2
# import numpy as np
# from azure.core.credentials import AzureKeyCredential
# from azure.ai.formrecognizer import DocumentAnalysisClient
# from msrest.authentication import CognitiveServicesCredentials
# import io
# from PIL import Image, ImageDraw
#
# # Azure OCR credentials
# ENDPOINT = 'https://chil-lab-ismail.cognitiveservices.azure.com/'
# API_KEY = 'Cgmj682Atpz4fm4eEzfD6gjxwtPtfP117gMFmAnaTUDQTarzEWgGJQQJ99AKACYeBjFXJ3w3AAALACOGg58W'
#
# # Initialize the Azure Computer Vision client
# cv_client = DocumentAnalysisClient(endpoint=ENDPOINT, credential=AzureKeyCredential(API_KEY))
#
#
# def extract_images_from_pdf(pdf_path):
#     """
#     Extract images from a PDF file using PyMuPDF.
#     Args:
#         pdf_path (str): Path to the PDF file.
#     Returns:
#         list: List of images as PIL.Image objects.
#     """
#     pdf_document = fitz.open(pdf_path)
#     images = []
#
#     for page_num in range(len(pdf_document)):
#         page = pdf_document[page_num]
#         for img_index, img in enumerate(page.get_images(full=True)):
#             xref = img[0]
#             base_image = pdf_document.extract_image(xref)
#             image_bytes = base_image["image"]
#             image = Image.open(io.BytesIO(image_bytes))
#             images.append((page_num, image))
#
#     return images
#
#
# def detect_curved_text(image, cv_client):
#     """
#     Detect text in an image using Azure Form Recognizer and return bounding boxes.
#     Args:
#         image (PIL.Image): Input image.
#         cv_client (DocumentAnalysisClient): Azure Form Recognizer client.
#     Returns:
#         list: List of bounding boxes [(x, y, x, y, x, y, x, y), ...].
#     """
#     # Convert image to bytes for Azure OCR
#     image_bytes = io.BytesIO()
#     image.save(image_bytes, format='JPEG')
#     image_bytes = image_bytes.getvalue()
#
#     # Analyze document
#     poller = cv_client.begin_analyze_document("prebuilt-read", document=image_bytes)
#     result = poller.result()
#
#     # Extract bounding boxes
#     bounding_boxes = []
#     for page in result.pages:
#         for line in page.lines:
#             bounding_boxes.append(line.bounding_box)  # Quadrilateral coordinates
#
#     return bounding_boxes
#
#
# def annotate_image_with_bounding_boxes(image, bounding_boxes):
#     """
#     Annotate an image with bounding boxes.
#     Args:
#         image (PIL.Image): Input image.
#         bounding_boxes (list): List of bounding boxes [(x, y, x, y, ...), ...].
#     Returns:
#         PIL.Image: Annotated image.
#     """
#     draw = ImageDraw.Draw(image)
#     for box in bounding_boxes:
#         points = [(box[i], box[i + 1]) for i in range(0, len(box), 2)]
#         draw.polygon(points, outline="red", width=3)  # Draw the quadrilateral
#     return image
#
#
# def process_pdf_for_curved_text(pdf_path, output_path):
#     """
#     Process a PDF to detect text and save an annotated version.
#     Args:
#         pdf_path (str): Path to the input PDF.
#         output_path (str): Path to save the annotated PDF.
#     """
#     images = extract_images_from_pdf(pdf_path)
#     annotated_pages = []
#
#     for page_num, image in images:
#         bounding_boxes = detect_curved_text(image, cv_client)
#         annotated_image = annotate_image_with_bounding_boxes(image.copy(), bounding_boxes)
#         annotated_pages.append((page_num, annotated_image))
#
#     # Save annotated images back to a PDF
#     pdf_document = fitz.open(pdf_path)
#     for page_num, annotated_image in annotated_pages:
#         page = pdf_document[page_num]
#         img_bytes = io.BytesIO()
#         annotated_image.save(img_bytes, format='PNG')
#         rect = fitz.Rect(0, 0, page.rect.width, page.rect.height)
#         page.insert_image(rect, stream=img_bytes.getvalue())
#
#     pdf_document.save(output_path)
#
#
# # Example usage
# pdf_path = r"C:\Users\ism_s\Desktop\RICE_Stuff\Comp490\Chil-Lab-Ballot-Project-main\Chil-Lab-Ballot-Project-main"
# output_path = "TESTING_Ballot.pdf"
# process_pdf_for_curved_text(pdf_path, output_path)
