import cv2
import numpy as np

"""
Detects lines using the OpenCV library
Inputs: image, an image. thresh, the thresh
Outputs: The image with the lines detected and drawn
"""
def open_cv_lines(image, thresh, min_horizontal_length=150, min_vertical_length=150):
    horriz_lines = []
    vertical_lines = []
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    detected_horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    detected_vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Find contours for horizontal and vertical lines
    cnts_horizontal = cv2.findContours(detected_horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_horizontal = cnts_horizontal[0] if len(cnts_horizontal) == 2 else cnts_horizontal[1]

    cnts_vertical = cv2.findContours(detected_vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_vertical = cnts_vertical[0] if len(cnts_vertical) == 2 else cnts_vertical[1]

    # Draw detected horizontal and vertical lines on the original image
    for c in cnts_horizontal:
        x, y, w, h = cv2.boundingRect(c)
        if w > min_horizontal_length:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue for horizontal lines
            horriz_lines.append((x, y, w, h))


    for c in cnts_vertical:
        x, y, w, h = cv2.boundingRect(c)
        if h > min_vertical_length:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Blue for vertical lines
            vertical_lines.append((x, y, w, h))
    
    return image, horriz_lines, vertical_lines


"""
Detects boxes using the OpenCV library
Inputs: image, an image. thresh, the thresh
Outputs: The image with the boxes detected and drawn
"""
def open_cv_boxes(image, thresh, min_horizontal_length=50, min_vertical_length=50):
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectanges = []

    # Loop through the contours to draw rectangles
    for contour in contours:
        # Get the bounding rectangle for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Check if the contour meets the minimum size requirements for rectangles
        if w > min_horizontal_length and h > min_vertical_length:
            # Draw a rectangle around the contour
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rectanges.append((x,y,w,h))

    # Return the original image with rectangles drawn on it
    return image, rectanges

"""
Code to add contrast to the images to make faight lines stick out
Inputs: Image, the image we are working with
Outputs: An image with added contrast 
"""
def contrast_image(image):
    lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img


"""
Detects lines and boxes given image bytes
Inputs: image_bytes, the bytes of the image
Outputs: The image with lines and boxes drawn with the locations
"""
def analyze_document_opencv(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    enhansed_image = contrast_image(image)
    # Convert to grayscale and apply threshold to get binary image
    gray = cv2.cvtColor(enhansed_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Return the original image with lines (blue) and rectangles (green) drawn on it
    image_lines, horriz_lines, vertical_lines = open_cv_lines(enhansed_image, thresh)
    image_combined, rectangles = open_cv_boxes(image_lines, thresh)
    return image_combined, rectangles, (horriz_lines, vertical_lines)