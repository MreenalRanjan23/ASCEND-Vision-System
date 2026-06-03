import cv2
import os

folder = "../dataset/raw"

for file in os.listdir(folder):

    path = os.path.join(folder, file)

    img = cv2.imread(path)

    if img is None:
        continue

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    print(
        file,
        "Sharpness:",
        round(score,2)
    )