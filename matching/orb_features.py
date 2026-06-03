import cv2

IMAGE_PATH = "../dataset/raw/PSP_010734_2000.jpg"

img = cv2.imread(IMAGE_PATH)

gray = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2GRAY
)

orb = cv2.ORB_create(
    nfeatures=1000
)

keypoints, descriptors = orb.detectAndCompute(
    gray,
    None
)

print("Number of keypoints:", len(keypoints))

output = cv2.drawKeypoints(
    img,
    keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

cv2.imshow(
    "ORB Features",
    output
)

cv2.waitKey(0)
cv2.destroyAllWindows()