import cv2
import numpy as np
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import streamlit as st
import os
from io import BytesIO
from PIL import Image

ENDPOINT = "https://byrneazure.cognitiveservices.azure.com/"
KEY = "GI9tvnay139DSCW91oDi2bRrNEwebaepoDlopN9PmLrzbGnlhLlXJQQJ99ALAC1i4TkXJ3w3AAALACOGDmGl"


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

"""Correctly sees stamps, we can change the paddings and sizes etc"""
# class BallotFeatureDetector:
#     def __init__(self):
#         self.stamp_padding = 20  # Padding around detected stamps
#         self.min_stamp_size = 50  # Minimum size of stamp
#         self.max_stamp_size = 200  # Maximum size of stamp
#         self.circularity_threshold = 0.5  # Threshold for considering something circular
#
#     def detect_features(self, image, azure_result=None):
#         """Detect and mask stamps in ballot images."""
#         height, width = image.shape[:2]
#         mask = np.ones((height, width), dtype=np.uint8) * 255
#         regions = {'ballot_images': []}
#
#         # Convert to grayscale for stamp detection
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#         # Get binary image
#         _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#
#         # Find contours
#         contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         stamp_regions = []
#         for contour in contours:
#             # Calculate basic properties
#             area = cv2.contourArea(contour)
#             perimeter = cv2.arcLength(contour, True)
#             x, y, w, h = cv2.boundingRect(contour)
#
#             # Skip if too small or too large
#             if not (self.min_stamp_size < w < self.max_stamp_size and
#                     self.min_stamp_size < h < self.max_stamp_size):
#                 continue
#
#             # Calculate circularity
#             if perimeter == 0:
#                 continue
#             circularity = 4 * np.pi * area / (perimeter * perimeter)
#
#             # Calculate density
#             rect_area = w * h
#             density = area / rect_area if rect_area > 0 else 0
#
#             # Check if the region looks like a stamp
#             if (circularity > self.circularity_threshold and
#                     0.3 < density < 0.9):  # Not too sparse or too dense
#
#                 # If Azure results available, check if region contains text
#                 contains_text = False
#                 if azure_result is not None:
#                     for page in azure_result.pages:
#                         for word in page.words:
#                             # Get word center
#                             word_x = sum(p[0] for p in word.polygon) / len(word.polygon)
#                             word_y = sum(p[1] for p in word.polygon) / len(word.polygon)
#
#                             # Check if word center is within or near stamp region
#                             if (x - 10 <= word_x <= x + w + 10 and
#                                     y - 10 <= word_y <= y + h + 10):
#                                 contains_text = True
#                                 break
#                         if contains_text:
#                             break
#
#                 if contains_text:
#                     pad = self.stamp_padding
#                     x1 = max(0, x - pad)
#                     y1 = max(0, y - pad)
#                     w = min(width - x1, w + 2 * pad)
#                     h = min(height - y1, h + 2 * pad)
#                     stamp_regions.append((x1, y1, w, h))
#
#         # Merge overlapping stamp regions
#         if stamp_regions:
#             merged_regions = self._merge_overlapping_regions(stamp_regions)
#             for region in merged_regions:
#                 regions['ballot_images'].append(region)
#                 x, y, w, h = region
#                 mask[y:y + h, x:x + w] = 0
#
#         # Apply mask to image
#         masked_image = image.copy()
#         masked_image[mask == 0] = 255
#
#         return regions, masked_image
#
#     def _merge_overlapping_regions(self, regions):
#         """Merge overlapping regions."""
#         if not regions:
#             return []
#
#         # Sort regions by x coordinate
#         regions = sorted(regions, key=lambda r: r[0])
#         merged = []
#         current = list(regions[0])
#
#         for region in regions[1:]:
#             x1, y1, w1, h1 = current
#             x2, y2, w2, h2 = region
#
#             # Check for overlap
#             overlap_x = (x1 <= x2 + w2) and (x2 <= x1 + w1)
#             overlap_y = (y1 <= y2 + h2) and (y2 <= y1 + h1)
#
#             if overlap_x and overlap_y:
#                 # Merge regions
#                 current[0] = min(x1, x2)
#                 current[1] = min(y1, y2)
#                 current[2] = max(x1 + w1, x2 + w2) - current[0]
#                 current[3] = max(y1 + h1, y2 + h2) - current[1]
#             else:
#                 merged.append(tuple(current))
#                 current = list(region)
#
#         merged.append(tuple(current))
#         return merged
#
#     def filter_detections(self, detections, feature_regions):
#         """Filter out detections that fall within masked regions."""
#         filtered = []
#
#         for detection in detections:
#             is_valid = True
#             if len(detection) >= 3:  # Ensure detection has coordinates
#                 x, y = detection[1], detection[2]
#
#                 for x1, y1, w, h in feature_regions['ballot_images']:
#                     if (x1 <= x <= x1 + w and y1 <= y <= y1 + h):
#                         is_valid = False
#                         break
#
#             if is_valid:
#                 filtered.append(detection)
#
#         return filtered


"""WORKS FOR STAMP DETECTION IN TOP LEFT but incorrectly sees O's as stamps too"""
# class BallotFeatureDetector:
#     def __init__(self):
#         self.stamp_size = (150, 150)  # Expected size of stamp region
#         self.stamp_padding = 20  # Padding around stamp
#         self.circularity_threshold = 0.6  # Minimum circularity for stamp detection
#
#     def detect_features(self, image, azure_result=None):
#         """Detect and mask the stamp in ballot images."""
#         height, width = image.shape[:2]
#         mask = np.ones((height, width), dtype=np.uint8) * 255
#         regions = {'ballot_images': []}
#
#         # Convert to grayscale for processing
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#         # Apply adaptive thresholding
#         thresh = cv2.adaptiveThreshold(
#             gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY_INV, 11, 2
#         )
#
#         # Find potential stamp contours
#         contours, _ = cv2.findContours(
#             thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#         )
#
#         stamp_region = None
#         max_circularity = 0
#
#         # Look for circular-like contours in the top-left region of the image
#         search_region = (int(width * 0.25), int(height * 0.25))  # Only look in top-left quarter
#
#         for contour in contours:
#             x, y, w, h = cv2.boundingRect(contour)
#
#             # Only consider contours in the top-left region
#             if x > search_region[0] or y > search_region[1]:
#                 continue
#
#             # Check if contour is roughly the right size for a stamp
#             if not (50 < w < 200 and 50 < h < 200):
#                 continue
#
#             # Calculate circularity
#             area = cv2.contourArea(contour)
#             perimeter = cv2.arcLength(contour, True)
#             if perimeter == 0:
#                 continue
#
#             circularity = 4 * np.pi * area / (perimeter * perimeter)
#
#             # Update if this is the most circular shape found
#             if circularity > max_circularity and circularity > self.circularity_threshold:
#                 max_circularity = circularity
#                 stamp_region = (x, y, w, h)
#
#         # If we found a stamp region, add padding and mask it
#         if stamp_region:
#             x, y, w, h = stamp_region
#             pad = self.stamp_padding
#
#             # Add stamp region with padding
#             x1 = max(0, x - pad)
#             y1 = max(0, y - pad)
#             x2 = min(width, x + w + pad)
#             y2 = min(height, y + h + pad)
#
#             regions['ballot_images'].append((x1, y1, x2 - x1, y2 - y1))
#             mask[y1:y2, x1:x2] = 0
#
#             # If Azure results are available, check for any nearby text that might be part of the stamp
#             if azure_result is not None:
#                 for page in azure_result.pages:
#                     for word in page.words:
#                         # Convert polygon points to numpy array
#                         poly_points = np.array([(int(p[0]), int(p[1])) for p in word.polygon], dtype=np.int32)
#
#                         # Get bounding box
#                         word_x, word_y, word_w, word_h = cv2.boundingRect(poly_points)
#
#                         # If word is near the stamp region, include it in the mask
#                         if abs(word_x - x) < pad * 2 and abs(word_y - y) < pad * 2:
#                             wx1 = max(0, word_x - pad)
#                             wy1 = max(0, word_y - pad)
#                             wx2 = min(width, word_x + word_w + pad)
#                             wy2 = min(height, word_y + word_h + pad)
#                             mask[wy1:wy2, wx1:wx2] = 0
#
#         # Apply mask to image
#         masked_image = image.copy()
#         masked_image[mask == 0] = 255
#
#         return regions, masked_image
#
#     def filter_detections(self, detections, feature_regions):
#         """Filter out detections that fall within masked regions."""
#         filtered = []
#
#         for detection in detections:
#             is_valid = True
#             if len(detection) >= 3:  # Ensure detection has coordinates
#                 x, y = detection[1], detection[2]
#
#                 for x1, y1, w, h in feature_regions['ballot_images']:
#                     if (x1 <= x <= x1 + w and y1 <= y <= y1 + h):
#                         is_valid = False
#                         break
#
#             if is_valid:
#                 filtered.append(detection)
#
#         return filtered

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


"""OLD"""
#
# import cv2
# import numpy as np
# from typing import List, Tuple, Dict
#
# class BallotFeatureDetector:
#     def __init__(self):
#         # Initialize feature detectors
#         self.orb = cv2.ORB_create()
#         self.barcode_detector = cv2.barcode_BarcodeDetector()
#
#     def detect_features(self, image: np.ndarray) -> Dict[str, List[Tuple]]:
#
#         # Detect various extraneous features in ballot papers and return their bounding regions.
#         #
#         # Args:
#         #     image: numpy array of the input image
#         #
#         # Returns:
#         #     Dictionary containing detected regions for different feature types
#
#         regions = {
#             'images': self._detect_images(image),
#             'arrows': self._detect_arrows(image),
#             'barcodes': self._detect_barcodes(image),
#             'signatures': self._detect_signatures(image)
#         }
#         return regions
#
#     def _detect_images(self, image: np.ndarray) -> List[Tuple]:
#         #Detect image regions using texture and gradient analysis.
#         # Convert to grayscale
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#         # Calculate local standard deviation to identify texture-rich regions
#         kernel_size = 5
#         local_std = cv2.blur((gray - cv2.blur(gray, (kernel_size, kernel_size)))**2,
#                             (kernel_size, kernel_size))
#
#         # Threshold to find regions with high texture
#         _, texture_mask = cv2.threshold(local_std, 30, 255, cv2.THRESH_BINARY)
#         texture_mask = texture_mask.astype(np.uint8)
#
#         # Find contours of texture regions
#         contours, _ = cv2.findContours(texture_mask, cv2.RETR_EXTERNAL,
#                                      cv2.CHAIN_APPROX_SIMPLE)
#
#         # Filter contours based on area and aspect ratio
#         image_regions = []
#         for contour in contours:
#             area = cv2.contourArea(contour)
#             if area < 100:  # Skip tiny regions
#                 continue
#             x, y, w, h = cv2.boundingRect(contour)
#             aspect_ratio = w / h
#             if 0.2 < aspect_ratio < 5:  # Reasonable aspect ratio for images
#                 image_regions.append((x, y, w, h))
#
#         return image_regions
#
#     def _detect_arrows(self, image: np.ndarray) -> List[Tuple]:
#         #Detect arrows using contour analysis and shape detection.
#         # Convert to grayscale and threshold
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#
#         # Find contours
#         contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#
#         arrow_regions = []
#         for contour in contours:
#             # Calculate contour properties
#             area = cv2.contourArea(contour)
#             if area < 50:  # Skip tiny contours
#                 continue
#
#             # Get rotated rectangle
#             rect = cv2.minAreaRect(contour)
#             box = cv2.boxPoints(rect)
#             box = np.int0(box)
#
#             # Check if the shape could be an arrow
#             # Arrows typically have specific aspect ratios and convexity
#             hull = cv2.convexHull(contour)
#             hull_area = cv2.contourArea(hull)
#             solidity = float(area) / hull_area
#
#             if 0.4 < solidity < 0.9:  # Arrow-like solidity
#                 x, y, w, h = cv2.boundingRect(contour)
#                 arrow_regions.append((x, y, w, h))
#
#         return arrow_regions
#
#     def _detect_barcodes(self, image: np.ndarray) -> List[Tuple]:
#         #Detect barcodes using OpenCV's barcode detector.
#         ok, decoded_info, decoded_type, corners = self.barcode_detector.detectAndDecode(image)
#
#         barcode_regions = []
#         if ok:
#             for corner_set in corners:
#                 if corner_set is not None:
#                     # Convert corners to bounding box
#                     x_coords = corner_set[:, 0]
#                     y_coords = corner_set[:, 1]
#                     x = int(min(x_coords))
#                     y = int(min(y_coords))
#                     w = int(max(x_coords) - x)
#                     h = int(max(y_coords) - y)
#                     barcode_regions.append((x, y, w, h))
#
#         return barcode_regions
#
#     def _detect_signatures(self, image: np.ndarray) -> List[Tuple]:
#         #Detect potential signature regions using ink density and stroke analysis.
#         # Convert to grayscale
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#         # Apply adaptive thresholding
#         thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                      cv2.THRESH_BINARY_INV, 11, 2)
#
#         # Find contours
#         contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
#                                      cv2.CHAIN_APPROX_SIMPLE)
#
#         signature_regions = []
#         for contour in contours:
#             area = cv2.contourArea(contour)
#             if area < 500:  # Skip tiny regions
#                 continue
#
#             x, y, w, h = cv2.boundingRect(contour)
#             aspect_ratio = w / h
#             extent = area / (w * h)
#
#             # Signatures typically have specific characteristics
#             if (2 < aspect_ratio < 7 and  # Wider than tall
#                 0.1 < extent < 0.4):      # Relatively sparse
#                 signature_regions.append((x, y, w, h))
#
#         return signature_regions
#
#     def filter_detections(self, detections: List[Tuple],
#                          feature_regions: Dict[str, List[Tuple]]) -> List[Tuple]:
#
#         # Filter out detections that fall within extraneous feature regions.
#         #
#         # Args:
#         #     detections: List of detected elements (words, boxes, lines)
#         #     feature_regions: Dictionary of detected feature regions
#         #
#         # Returns:
#         #     Filtered list of detections
#
#         def _is_within_region(detection, region):
#             dx, dy, dw, dh = detection
#             rx, ry, rw, rh = region
#
#             # Check if detection center falls within region
#             d_center_x = dx + dw/2
#             d_center_y = dy + dh/2
#
#             return (rx <= d_center_x <= rx + rw and
#                    ry <= d_center_y <= ry + rh)
#
#         filtered_detections = []
#         for detection in detections:
#             is_valid = True
#
#             # Check against all feature regions
#             for regions in feature_regions.values():
#                 for region in regions:
#                     if _is_within_region(detection, region):
#                         is_valid = False
#                         break
#                 if not is_valid:
#                     break
#
#             if is_valid:
#                 filtered_detections.append(detection)
#
#         return filtered_detections
#
#
#
"""OLD THING"""
# import cv2
# import numpy as np
# from azure.core.credentials import AzureKeyCredential
# from azure.ai.formrecognizer import DocumentAnalysisClient
# import streamlit as st
# import os
#
# # Direct credential assignment (for development)
# ENDPOINT = "https://chil-lab-ismail.cognitiveservices.azure.com/"
# KEY = "Cgmj682Atpz4fm4eEzfD6gjxwtPtfP117gMFmAnaTUDQTarzEWgGJQQJ99AKACYeBjFXJ3w3AAALACOGg58W"
#
# # define directories
# ballots_folder = "ballots"
# temp_folder = "temp"
#
# # Initialize the client directly with the credentials
# document_analysis_client = DocumentAnalysisClient(
#     endpoint=ENDPOINT,
#     credential=AzureKeyCredential(KEY)
# )
#
#
# def analyze_document(content):
#     """Analyze document using Azure Form Recognizer"""
#     poller = document_analysis_client.begin_analyze_document(
#         "prebuilt-layout", content)
#     return poller.result()
#
#
# def display_annotated_image(image_bytes, analyze_result):
#     """Display image with annotations for words and selection marks"""
#     image = cv2.imdecode(np.frombuffer(
#         image_bytes, np.uint8), cv2.IMREAD_COLOR)
#
#     # Create a copy to avoid modifying the original
#     annotated_image = image.copy()
#
#     for page in analyze_result.pages:
#         for word_info in page.words:
#             pts = np.array(word_info.polygon, np.int32).reshape((-1, 1, 2))
#             annotated_image = cv2.polylines(annotated_image, [pts], True, (0, 255, 0), 2)
#         for selection_mark in page.selection_marks:
#             selection_pts = np.array(selection_mark.polygon, np.int32).reshape(
#                 (-1, 1, 2))
#             annotated_image = cv2.polylines(annotated_image, [selection_pts], True, (0, 0, 255), 2)
#
#     return annotated_image
#
#
# # If you want to use environment variables or secrets in production:
# def get_credentials():
#     """Get credentials from environment variables or secrets"""
#     try:
#         # Try to get from streamlit secrets
#         endpoint = st.secrets.get("FORM_RECOGNIZER_ENDPOINT", ENDPOINT)
#         key = st.secrets.get("FORM_RECOGNIZER_KEY", KEY)
#     except Exception:
#         # Fallback to environment variables
#         endpoint = os.getenv("FORM_RECOGNIZER_ENDPOINT", ENDPOINT)
#         key = os.getenv("FORM_RECOGNIZER_KEY", KEY)
#
#     return endpoint, key
#
#
# # Optional: Initialize client with credentials from environment/secrets
# def initialize_client():
#     """Initialize the Form Recognizer client with credentials"""
#     endpoint, key = get_credentials()
#     return DocumentAnalysisClient(
#         endpoint=endpoint,
#         credential=AzureKeyCredential(key)
#     )
#
#
# # For testing the module directly
# if __name__ == "__main__":
#     print("Azure Form Recognizer utility initialized successfully")