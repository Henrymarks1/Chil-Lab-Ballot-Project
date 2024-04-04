from oauth2client.service_account import ServiceAccountCredentials
import gspread

# Set up Google Sheets API
scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('ballot-parsing-865d0dd8b2a8.json', scope)
client = gspread.authorize(creds)

sheet = client.open('CHIL Ballot Data').sheet1
sheet.clear()
data = [["Word", "Center_X", "Center_Y", "Length", "Height"]]


def extract_words_and_coordinates(analyze_result):
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

            data.append([word_text, center_x, center_y, length, height])


def write_excel():
    range = f'A1:E{len(data)}'
    sheet.update(range, data)