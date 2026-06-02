# code used for thia project

import cv2
import numpy as np
import requests
from datetime import datetime
import pytesseract

# Configure these variables
ESP32_CAM_URL = "http://172.20.10.4/?res=10"
OUTPUT_FILE = "raw.txt"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Set up Tesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    contours, _ = cv2.findContours(
        edged.copy(),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )[:10]

    plate_contour = None

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(
            contour,
            0.018 * perimeter,
            True
        )

        if len(approx) == 4:
            plate_contour = approx
            break

    if plate_contour is not None:
        mask = np.zeros(gray.shape, np.uint8)

        cv2.drawContours(
            mask,
            [plate_contour],
            0,
            255,
            -1
        )

        cv2.bitwise_and(frame, frame, mask=mask)

        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))

        cropped = gray[
            topx:bottomx + 1,
            topy:bottomy + 1
        ]

        text = pytesseract.image_to_string(
            cropped,
            config="--psm 11"
        )

        text = ''.join(
            e for e in text if e.isalnum()
        ).upper()

        if len(text) > 4:
            return text

    return None


def save_to_file(plate_text):
    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(OUTPUT_FILE, "a") as f:
        f.write(
            f"{timestamp} - {plate_text}\n"
        )


def main():
    stream = requests.get(
        ESP32_CAM_URL,
        stream=True
    )

    bytes_data = bytes()

    while True:
        bytes_data += stream.raw.read(1024)

        a = bytes_data.find(b'\xff\xd8')
        b = bytes_data.find(b'\xff\xd9')

        if a != -1 and b != -1:
            jpg = bytes_data[a:b + 2]
            bytes_data = bytes_data[b + 2:]

            frame = cv2.imdecode(
                np.frombuffer(
                    jpg,
                    dtype=np.uint8
                ),
                cv2.IMREAD_COLOR
            )

            if frame is not None:
                plate_text = process_frame(frame)

                if plate_text:
                    save_to_file(plate_text)

                cv2.imshow(
                    "ESP32-CAM Stream",
                    frame
                )

                if cv2.waitKey(1) == 27:
                    break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
