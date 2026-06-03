import cv2

# =========================
# Load Images
# =========================

img1 = cv2.imread(
    "../dataset/raw/PSP_010734_2000.jpg"
)

img2 = cv2.imread(
    "../dataset/raw/PSP_010734_2000.jpg"
)

# =========================
# ORB Feature Extraction
# =========================

orb = cv2.ORB_create(
    nfeatures=1000
)

kp1, des1 = orb.detectAndCompute(
    img1,
    None
)

kp2, des2 = orb.detectAndCompute(
    img2,
    None
)

print("Image 1 Keypoints:", len(kp1))
print("Image 2 Keypoints:", len(kp2))

# =========================
# KNN Matcher
# =========================

bf = cv2.BFMatcher(
    cv2.NORM_HAMMING
)

matches = bf.knnMatch(
    des1,
    des2,
    k=2
)

# =========================
# Lowe's Ratio Test
# =========================

good_matches = []

for m, n in matches:

    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print("Good Matches:", len(good_matches))

# =========================
# Draw Matches
# =========================

result = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    good_matches[:50],   # draw best 50 matches
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# =========================
# Enlarge for Visibility
# =========================

scale = 4

h, w = result.shape[:2]

result_large = cv2.resize(
    result,
    (w * scale, h * scale),
    interpolation=cv2.INTER_NEAREST
)

cv2.imshow(
    "ORB Matches - Lowe Ratio Test",
    result_large
)

cv2.waitKey(0)
cv2.destroyAllWindows()