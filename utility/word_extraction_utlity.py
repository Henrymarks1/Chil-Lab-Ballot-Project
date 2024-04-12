import pandas as pd

def create_downloadable_dataframe(word_locations, boxes, horriz_lines, vert_lines):
    """Create a pandas DataFrame from the data and return a CSV string."""
    # Prepare the data as a list of dictionaries, which pandas can easily convert to a DataFrame
    data = []
    data.extend([{'Type': 'Word Location', 'Data': loc} for loc in word_locations])
    data.extend([{'Type': 'Box', 'Data': box} for box in boxes])
    data.extend([{'Type': 'Horriontal Line', 'Data': horriz_line} for horriz_line in horriz_lines])
    data.extend([{'Type': 'Vertical Line', 'Data': vert_lines} for vert_lines in vert_lines])

    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Convert DataFrame to CSV string
    return df.to_csv(index=False)


"""
Extract the words from azure page
Input: the results from the azure call
Output: The words and their resepective locations
"""
def extract_words_and_coordinates(analyze_result):
    word_locations = []
    for page in analyze_result.pages:
        for word in page.words:
            word_text = word.content
            x1, y1 = word.polygon[0]
            x2, y2 = word.polygon[1]
            x3, y3 = word.polygon[2]
            x4, y4 = word.polygon[3]

            center_x = (x1 + x2 + x3 + x4) / 4
            center_y = (y1 + y2 + y3 + y4) / 4
            length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
            height = ((x4 - x1)**2 + (y4 - y1)**2)**0.5

            word_locations.append([word_text, center_x, center_y, length, height])
    return word_locations


# def write_excel():
#     range = f'A1:E{len(data)}'
#     sheet.update(range, data)