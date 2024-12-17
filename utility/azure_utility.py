import cv2
import numpy as np
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import streamlit as st
import os
from io import BytesIO
from PIL import Image

ENDPOINT = "https://byrneazure.cognitiveservices.azure.com/"
KEY = "GI9tvnay139DSCW91oDi2bRrNEwebaepoD1opN9PmLrzbGnlhLIXJQQJ99ALAC1i4TkXJ3w3AAALACOGDmGI"


def compress_image(image_bytes, max_size_mb=4.0):
    """Compress image to ensure it's under Azure's size limit"""
    max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes

    # Convert bytes to PIL Image
    image = Image.open(BytesIO(image_bytes))

    # Convert to RGB if necessary
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')

    # Start with original size and quality
    quality = 95
    output = BytesIO()
    image.save(output, format='JPEG', quality=quality)

    # Reduce quality until file size is under limit
    while output.tell() > max_size_bytes and quality > 50:
        output = BytesIO()
        image.save(output, format='JPEG', quality=quality)
        quality -= 5

    # If still too large, resize the image
    if output.tell() > max_size_bytes:
        while output.tell() > max_size_bytes:
            width, height = image.size
            image = image.resize((int(width * 0.75), int(height * 0.75)), Image.Resampling.LANCZOS)
            output = BytesIO()
            image.save(output, format='JPEG', quality=quality)

    return output.getvalue()


document_analysis_client = DocumentAnalysisClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(KEY)
)


def analyze_document(content):
    """Analyze document using Azure Form Recognizer with automatic size handling"""
    try:
        # Compress image if needed
        compressed_content = compress_image(content)

        # Analyze with Azure
        poller = document_analysis_client.begin_analyze_document(
            "prebuilt-layout", compressed_content)
        return poller.result()
    except Exception as e:
        st.error(f"Error in analyze_document: {str(e)}")
        raise

"""Separated Ballot and Stamp Detector for better modularity"""
class StampDetector:
    def __init__(self):
        self.padding = 20  # Padding around detected stamps
        self.min_size = 50  # Minimum size of stamp
        self.max_size = 200  # Maximum size of stamp
        self.circularity_threshold = 0.5  # Threshold for considering something circular
        self.density_range = (0.3, 0.9)  # Valid density range for stamps
        self.text_margin = 10  # Margin for checking text near stamp

    def detect_stamps(self, image, azure_result=None):
        """Detect stamp regions in an image."""
        height, width = image.shape[:2]

        # Convert to grayscale for stamp detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find potential stamp regions
        stamp_regions = []
        for contour in contours:
            if self._is_stamp_contour(contour, width, height):
                x, y, w, h = cv2.boundingRect(contour)

                # Check for text if Azure results available
                if azure_result and self._contains_text(x, y, w, h, azure_result):
                    # Add padding to region
                    pad = self.padding
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    w = min(width - x1, w + 2 * pad)
                    h = min(height - y1, h + 2 * pad)
                    stamp_regions.append((x1, y1, w, h))

        # Merge overlapping regions
        return self._merge_overlapping_regions(stamp_regions)

    def _is_stamp_contour(self, contour, width, height):
        """Check if a contour has stamp-like properties."""
        # Get basic properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)

        # Check size constraints
        if not (self.min_size < w < self.max_size and
                self.min_size < h < self.max_size):
            return False

        # Check circularity
        if perimeter == 0:
            return False
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity <= self.circularity_threshold:
            return False

        # Check density
        rect_area = w * h
        if rect_area == 0:
            return False
        density = area / rect_area
        min_density, max_density = self.density_range
        if not (min_density < density < max_density):
            return False

        return True

    def _contains_text(self, x, y, w, h, azure_result):
        """Check if a region contains text."""
        for page in azure_result.pages:
            for word in page.words:
                # Get word center
                word_x = sum(p[0] for p in word.polygon) / len(word.polygon)
                word_y = sum(p[1] for p in word.polygon) / len(word.polygon)

                # Check if word center is within or near region
                if (x - self.text_margin <= word_x <= x + w + self.text_margin and
                        y - self.text_margin <= word_y <= y + h + self.text_margin):
                    return True
        return False

    def _merge_overlapping_regions(self, regions):
        """Merge overlapping regions."""
        if not regions:
            return []

        # Sort regions by x coordinate
        regions = sorted(regions, key=lambda r: r[0])
        merged = []
        current = list(regions[0])

        for region in regions[1:]:
            x1, y1, w1, h1 = current
            x2, y2, w2, h2 = region

            # Check for overlap
            overlap_x = (x1 <= x2 + w2) and (x2 <= x1 + w1)
            overlap_y = (y1 <= y2 + h2) and (y2 <= y1 + h1)

            if overlap_x and overlap_y:
                # Merge regions
                current[0] = min(x1, x2)
                current[1] = min(y1, y2)
                current[2] = max(x1 + w1, x2 + w2) - current[0]
                current[3] = max(y1 + h1, y2 + h2) - current[1]
            else:
                merged.append(tuple(current))
                current = list(region)

        merged.append(tuple(current))
        return merged

class BallotFeatureDetector:
    def __init__(self):
        self.stamp_detector = StampDetector()

    def detect_features(self, image, azure_result=None):
        """Detect and mask stamps in ballot images."""
        height, width = image.shape[:2]
        mask = np.ones((height, width), dtype=np.uint8) * 255
        regions = {'ballot_images': []}

        # Detect stamps
        stamp_regions = self.stamp_detector.detect_stamps(image, azure_result)

        # Add regions and create mask
        for region in stamp_regions:
            regions['ballot_images'].append(region)
            x, y, w, h = region
            mask[y:y + h, x:x + w] = 0

        # Apply mask to image
        masked_image = image.copy()
        masked_image[mask == 0] = 255

        return regions, masked_image

    def filter_detections(self, detections, feature_regions):
        """Filter out detections that fall within masked regions."""
        filtered = []

        for detection in detections:
            is_valid = True
            if len(detection) >= 3:  # Ensure detection has coordinates
                x, y = detection[1], detection[2]

                for x1, y1, w, h in feature_regions['ballot_images']:
                    if (x1 <= x <= x1 + w and y1 <= y <= y1 + h):
                        is_valid = False
                        break

            if is_valid:
                filtered.append(detection)

        return filtered



def display_annotated_image(image_bytes, analyze_result):
    """Display image with annotations for words and selection marks"""
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Create a copy to avoid modifying the original
    annotated_image = image.copy()

    for page in analyze_result.pages:
        for word_info in page.words:
            pts = np.array(word_info.polygon, np.int32).reshape((-1, 1, 2))
            annotated_image = cv2.polylines(annotated_image, [pts], True, (0, 255, 0), 2)
        for selection_mark in page.selection_marks:
            selection_pts = np.array(selection_mark.polygon, np.int32).reshape((-1, 1, 2))
            annotated_image = cv2.polylines(annotated_image, [selection_pts], True, (0, 0, 255), 2)

    return annotated_image


def get_credentials():
    """Get credentials from environment variables or secrets"""
    try:
        endpoint = st.secrets.get("FORM_RECOGNIZER_ENDPOINT", ENDPOINT)
        key = st.secrets.get("FORM_RECOGNIZER_KEY", KEY)
    except Exception:
        endpoint = os.getenv("FORM_RECOGNIZER_ENDPOINT", ENDPOINT)
        key = os.getenv("FORM_RECOGNIZER_KEY", KEY)
    return endpoint, key

