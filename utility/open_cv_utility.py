import cv2
import numpy as np

"""
Purpose: Detects horizontal and vertical lines in an image (in this case, PDF)
using the OpenCV library. The function modifies the input image in-place by
drawing lines on it.

Inputs: 
- image: the input image on which to detect lines. 
- thresh: a pre-processed binary image (thresholded image)

Outputs: 
- image: the original image with detected horizontal and vertical lines drawn on it 
- horriz_lines: a list of tuples where each tuple represents a detected horizontal line 
- vertical_line: a list of tuples where each tuple represents a detected vertical line 

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
    
    # Below is the code for dotted/faint lines...

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Experiment with different kernel sizes to enhance the dotted lines
    kernel1 = np.ones((2, 5), np.uint8)  # Experimented with a smaller kernel... may work better? Try more
    kernel2 = np.ones((9, 9), np.uint8) 

    # Use the morphological gradient which is the difference between dilation and erosion
    # This can sometimes enhance the outline of the dots more effectively
    img_gradient = cv2.morphologyEx(img_thresh, cv2.MORPH_GRADIENT, kernel1)

    # Apply a slight blur which can help to join the dots in the dotted lines
    img_blur = cv2.GaussianBlur(img_gradient, (3, 3), 0)

    # Apply Hough Lines again with adjusted parameters
    img_lines = cv2.HoughLinesP(img_blur, 10, np.pi/180, threshold=20, minLineLength=440, maxLineGap=15)
    
    # Draw detected dotted lines on the original image
    if img_lines is not None:
        for line in img_lines:
            for x1, y1, x2, y2 in line:
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for dotted lines
    
    # Return the updated image with both solid and dotted lines drawn
    return image, horriz_lines, vertical_lines

"""
Purpose: Detects boxes in an image (in this case, PDF) using the OpenCV library.
The function modifies the input image in-place by drawing boxes on it.

Inputs: 
- image: the input image on which to detect lines. 
- thresh: a pre-processed binary image (thresholded image)

Outputs: 
- image: the original image with detected horizontal and vertical lines drawn on it
- horriz_lines: a list of tuples where each tuple represents a detected horizontal line 
- vertical_line: a list of tuples where each tuple represents a detected vertical line 

"""
def open_cv_boxes(image, thresh, min_horizontal_length=50, min_vertical_length=50):
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
Purpose: Intensifies the contrast of the image to attempt to detect faint lines
and other relevant components of the PDF file.

Inputs: 
- image: the input image on which to increase contrast.

Outputs: 
- image: the original image with intensified contrast.

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
Purpose: Detects lines and boxes in an image (in this case, PDF) using the
OpenCV library. The function modifies the input image in-place by drawing
detected lines and boxes on it.

Inputs: 
- image_bytes: the bytes of the input image.

Outputs: 
- image: the original image with detected lines and boxes drawn on it.

"""
def analyze_document_opencv(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    enhanced_image = contrast_image(image)
    # Convert to grayscale and apply threshold to get binary image
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Return the original image with lines (blue) and rectangles (green) drawn on it
    image_lines, horriz_lines, vertical_lines = open_cv_lines(enhanced_image, thresh)
    image_combined, rectangles = open_cv_boxes(image_lines, thresh)
    return image_combined, rectangles, horriz_lines, vertical_lines