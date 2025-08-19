from PIL import Image
import io
from base64 import b64encode

def preprocess_image(image_path: str) -> str:
    image = Image.open(image_path)
    image = image.resize((300, 300))
    image = image.convert("RGB")

    img_bytes_arr = io.BytesIO()
    image.save(img_bytes_arr, format="JPEG")
    img_bytes = img_bytes_arr.getvalue()
    b64_image = b64encode(img_bytes).decode("utf-8")

    return b64_image
