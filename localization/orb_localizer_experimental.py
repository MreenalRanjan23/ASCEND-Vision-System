import cv2

# ==========================
# Load Images
# ==========================

reference = cv2.imread(
    "../dataset/reference/target.jpg"
)

cap = cv2.VideoCapture(1)

# ==========================
# ORB
# ==========================

orb = cv2.ORB_create(
    nfeatures=2000
)

kp1, des1 = orb.detectAndCompute(
    reference,
    None
)

bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=True
)

# ==========================
# Loop
# ==========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    kp2, des2 = orb.detectAndCompute(
        frame,
        None
    )

    if des2 is not None:

        matches = bf.match(
            des1,
            des2
        )

        matches = sorted(
            matches,
            key=lambda x: x.distance
        )

        good_matches = matches[:50]

        cv2.putText(
            frame,
            f"Matches: {len(good_matches)}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    cv2.imshow(
        "ORB Localization",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()