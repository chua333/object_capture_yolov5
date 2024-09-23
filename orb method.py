import cv2
import numpy as np
import os

# Load template image
template_path = 'second.jpg'
template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
template_kp, template_des = None, None

# Initialize ORB detector
orb = cv2.ORB_create()

# Detect keypoints and descriptors in the template
template_kp, template_des = orb.detectAndCompute(template, None)

# FLANN based matcher parameters
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)  # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)

# Specify video source and output directory
video_path = input("Enter the video source file path: ")
output_dir = input("Enter the output directory to save frames: ")
os.makedirs(output_dir, exist_ok=True)

# Open video file
cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors in the frame
    frame_kp, frame_des = orb.detectAndCompute(frame_gray, None)

    # Match descriptors
    matches = flann.knnMatch(template_des, frame_des, k=2)

    # Apply ratio test
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    # Define threshold for good matches
    MIN_MATCH_COUNT = 10

    if len(good_matches) > MIN_MATCH_COUNT:
        # Get the keypoint coordinates of the good matches
        src_pts = np.float32([template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find the homography matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Get the dimensions of the template
        h, w = template.shape

        # Get the coordinates of the corners of the template in the frame
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        # Draw the bounding box
        frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

        # Save the frame with bounding box
        frame_path = os.path.join(output_dir, f'frame_{frame_count}.jpg')
        cv2.imwrite(frame_path, frame)
        print(f"Saved frame {frame_count} with detected template to {frame_path}")

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
