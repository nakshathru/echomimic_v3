import numpy as np
from PIL import Image
import cv2

# Use OpenCV's built-in Haar cascade for face detection
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def _to_numpy(image):
    """Normalize input to a HxWxC numpy array (RGB)."""
    # Case 1: string path
    if isinstance(image, str):
        pil_img = Image.open(image).convert("RGB")
        return np.array(pil_img)

    # Case 2: PIL image
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))

    # Case 3: already numpy
    arr = np.array(image)
    # If grayscale, expand to 3 channels
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    # If RGBA, drop alpha
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def get_mask_coord(image):
    """
    Return a face bounding box for EchoMimicV3.

    Input:
        image: file path (str), PIL.Image, or numpy array

    Output:
        y1, y2, x1, x2, h, w
    """
    img = _to_numpy(image)
    h, w = img.shape[:2]

    # Convert to grayscale for detection
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(int(min(h, w) * 0.1), int(min(h, w) * 0.1)),
    )

    if len(faces) > 0:
        # Use the largest detected face
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + fw)
        y2 = min(h, y + fh)

        # Pad a bit to include more of the head
        pad_y = int(0.1 * fh)
        pad_x = int(0.1 * fw)
        y1 = max(0, y1 - pad_y)
        y2 = min(h, y2 + pad_y)
        x1 = max(0, x1 - pad_x)
        x2 = min(w, x2 + pad_x)
    else:
        # Fallback: central region
        size = int(min(h, w) * 0.6)
        cy, cx = h // 2, w // 2
        y1 = max(0, cy - size // 2)
        y2 = min(h, cy + size // 2)
        x1 = max(0, cx - size // 2)
        x2 = min(w, cx + size // 2)

    # Safety
    if y2 <= y1 or x2 <= x1:
        y1, y2, x1, x2 = 0, h, 0, w

    return y1, y2, x1, x2, h, w
