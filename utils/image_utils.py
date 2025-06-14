from PIL import Image
import base64
import io
import pytesseract

def extract_text_from_image(base64_image):
    image_data = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image_data))
    return pytesseract.image_to_string(image)
