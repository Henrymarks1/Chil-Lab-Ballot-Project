from io import BytesIO
import os

from pdf2image import convert_from_bytes, convert_from_path
ballots_folder = "ballots"

'''
Purpose: Passes in and loads all of the input PDF ballots.

Inputs: 
N/A

Outputs: 
example_images: contains the paths to all the files in the specified directory that have the correct file extensions.
'''
def load_example_images():
    example_images = []
    for filename in os.listdir(ballots_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', 'pdf')):
            file_path = os.path.join(ballots_folder, filename)
            example_images.append(file_path)
    return example_images


'''
Purpose: Handles the file path of the PDFs. Determines whether or not the file
is an existing PDF Ballot sample on the program or 2) a new, uploaded file.
Later recognizes all of the images.

Inputs: N/A

Outputs: byte_images: a list containing byte arrays of all images extracted and
converted from the input
'''
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
