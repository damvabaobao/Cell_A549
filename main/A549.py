import cv2
import numpy as np
import matplotlib.pyplot as plt
from roboflow import KMeans

SCALE_BAR_UM = 50 # Chinh theo anh

# Model

rf = Roboflow(api_key="_____")
project = rf.workspace().project("______")
model = project.version("___").model

# Load Model

image_path = r"______"
image = cv2.imread(image_path)

if image is None:
        print("Khong load duoc anh!!!!")
        exit()

# SCALE BAR
def dectect_scale_bar(image):
        gray = cv2.cvtClor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if h > 0 and w/h > 5 and w > 50:
                        candidates.append((x, y, w, h))

        return max(candidates, key=lambda x : x[2]) if candidates else None
scale = dectect_scale_bar(image)
if scale:
        x, y, w, h = scale
        scale_bar_pixels = w
else :
        print("Fallback scale")
        scale_bar_pixels = 200
um_per_pixel = SCALE_BAR_UM / scale_bar_pixels
are_convert = um_per_pixel ** 2

