"""this scales an image if needed. It was used when we were trying to simply 
find all the images within the pdf, but we no longer use it """

# import cv2
# import numpy as np
# from PIL import Image
# from io import BytesIO
#
#
#
# class ImageProcessor:
#     def __init__(self, max_dimension=4000, target_size_mb=3.5):
#         self.max_dimension = max_dimension
#         self.target_size_mb = target_size_mb * 1024 * 1024  # Convert MB to bytes
#
#     def resize_if_needed(self, image_bytes):
#         """Resize image if it exceeds maximum dimensions or file size."""
#         # Convert bytes to numpy array
#         nparr = np.frombuffer(image_bytes, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#
#         if img is None:
#             raise ValueError("Failed to decode image")
#
#         # Check dimensions
#         height, width = img.shape[:2]
#         if height > self.max_dimension or width > self.max_dimension:
#             # Calculate new dimensions
#             if height > width:
#                 new_height = self.max_dimension
#                 new_width = int(width * (self.max_dimension / height))
#             else:
#                 new_width = self.max_dimension
#                 new_height = int(height * (self.max_dimension / width))
#
#             img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
#
#         # Check file size and compress if needed
#         success, buffer = cv2.imencode('.png', img)
#         if not success:
#             raise ValueError("Failed to encode image")
#
#         if len(buffer) > self.target_size_mb:
#             quality = 95
#             while len(buffer) > self.target_size_mb and quality > 50:
#                 success, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
#                 quality -= 5
#
#         return buffer.tobytes()
#
#     def preprocess_for_feature_detection(self, img):
#         """Enhance image for better feature detection."""
#         # Convert to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#         # Apply adaptive histogram equalization
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         enhanced = clahe.apply(gray)
#
#         # Denoise
#         denoised = cv2.fastNlMeansDenoising(enhanced)
#
#         # Adaptive thresholding
#         thresh = cv2.adaptiveThreshold(
#             denoised,
#             255,
#             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY_INV,
#             11,
#             2
#         )
#
#         return thresh, enhanced