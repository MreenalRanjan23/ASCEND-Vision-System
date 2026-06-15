import os
import sys
import cv2

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from feature_matching.dino_verifier import verify

# =====================================
# Test Image
# =====================================

IMAGE_PATH = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
    "test_images",
    "test.jpg"
)

# =====================================
# Load Image
# =====================================

image = cv2.imread(
    IMAGE_PATH
)

if image is None:

    print(
        "Could not load image"
    )

    exit()

# =====================================
# Verify
# =====================================

result = verify(
    image
)

# =====================================
# Output
# =====================================

print("\nVerification Result\n")

print(
    f"Match      : {result['match']}"
)

print(
    f"Similarity : {result['similarity']:.4f}"
)