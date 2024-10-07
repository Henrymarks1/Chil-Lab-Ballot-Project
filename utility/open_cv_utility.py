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

    cnts_horizontal = cv2.findContours(detected_horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_horizontal = cnts_horizontal[0] if len(cnts_horizontal) == 2 else cnts_horizontal[1]

    cnts_vertical = cv2.findContours(detected_vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_vertical = cnts_vertical[0] if len(cnts_vertical) == 2 else cnts_vertical[1]

    for c in cnts_horizontal:
        x, y, w, h = cv2.boundingRect(c)
        if w > min_horizontal_length:
            horriz_lines.append((x, y, w, h))

    for c in cnts_vertical:
        x, y, w, h = cv2.boundingRect(c)
        if h > min_vertical_length:
            vertical_lines.append((x, y, w, h))

    return horriz_lines, vertical_lines

    
    # # Below is the code for dotted/faint lines...

    # img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # # Experiment with different kernel sizes to enhance the dotted lines
    # kernel1 = np.ones((2, 5), np.uint8)  # Experimented with a smaller kernel... may work better? Try more
    # kernel2 = np.ones((9, 9), np.uint8) 

    # # Use the morphological gradient which is the difference between dilation and erosion
    # # This can sometimes enhance the outline of the dots more effectively
    # img_gradient = cv2.morphologyEx(img_thresh, cv2.MORPH_GRADIENT, kernel1)

    # # Apply a slight blur which can help to join the dots in the dotted lines
    # img_blur = cv2.GaussianBlur(img_gradient, (3, 3), 0)

    # # Apply Hough Lines again with adjusted parameters
    # img_lines = cv2.HoughLinesP(img_blur, 10, np.pi/180, threshold=20, minLineLength=440, maxLineGap=15)
    
    # # Draw detected dotted lines on the original image
    # if img_lines is not None:
    #     for line in img_lines:
    #         for x1, y1, x2, y2 in line:
    #             cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for dotted lines
    
    # Return the updated image with both solid and dotted lines drawn
    #return image, horriz_lines, vertical_lines

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
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > min_horizontal_length and h > min_vertical_length:
            rectangles.append((x, y, w, h))

    return rectangles

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
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    horriz_lines, vertical_lines = open_cv_lines(enhanced_image, thresh)
    rectangles = open_cv_boxes(enhanced_image, thresh)

    # Filter out lines that intersect with boxes
    filtered_horriz_lines = filter_lines(horriz_lines, rectangles)
    filtered_vertical_lines = filter_lines(vertical_lines, rectangles)

    # Draw the filtered lines and boxes on the image
    for x, y, w, h in filtered_horriz_lines:
        cv2.rectangle(enhanced_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for x, y, w, h in filtered_vertical_lines:
        cv2.rectangle(enhanced_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    for x, y, w, h in rectangles:
        cv2.rectangle(enhanced_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return enhanced_image, rectangles, filtered_horriz_lines, filtered_vertical_lines

"""
Purpose: Detects if the lines intersect with the boxes
Inputs: 
 - line: a vertical or horizontal line represented by a tuple 
 - box:  a list of tuples that represent a rectangle

Outputs: 
- boolean: true if the line intersects, false otherwise

"""

def line_intersects_box(line, box):
    x1, y1, w1, h1 = line
    x2, y2, w2, h2 = box

    # Check if the line is completely inside the box
    if x1 >= x2 and x1 + w1 <= x2 + w2 and y1 >= y2 and y1 + h1 <= y2 + h2:
        return True

    # Check if the line intersects any of the box's edges
    line_left, line_right = x1, x1 + w1
    line_top, line_bottom = y1, y1 + h1
    box_left, box_right = x2, x2 + w2
    box_top, box_bottom = y2, y2 + h2

    if (line_left <= box_right and line_right >= box_left and
            line_top <= box_bottom and line_bottom >= box_top):
        return True

    return False


"""
Purpose: 
Inputs: 
 - line: a list of tuples where each tuple represents a detected horizontal or vertical line
 - box:  a list of tuples where each tuple represents each line in a rectangle

Outputs: 
- filtered_lines: returns a list of filtered lines (lines that are double counted)

"""

def filter_lines(lines, boxes):
    filtered_lines = []
    for line in lines:
        if not any(line_intersects_box(line, box) for box in boxes):
            filtered_lines.append(line)
    return filtered_lines


