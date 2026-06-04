import cv2

for index in range(5):

    print(f"\nOpening Camera {index}")

    cap = cv2.VideoCapture(index)

    if not cap.isOpened():

        print("Not Opened")
        continue

    while True:

        ret, frame = cap.read()

        if not ret:
            print("No Frame")
            break

        cv2.imshow(
            f"Camera {index}",
            frame
        )

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()