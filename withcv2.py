import torch
import cv2
import os
from skimage.metrics import structural_similarity as ssim


# function to compute similarity using CLIP based
def compute_clip(img1, img2):
    import torch
    import open_clip
    from sentence_transformers import util
    from PIL import Image

    # Image processing model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
    model.to(device)

    def imageEncoder(img):
        img1 = Image.fromarray(img).convert('RGB')
        img1 = preprocess(img1).unsqueeze(0).to(device)
        img1 = model.encode_image(img1)
        return img1

    def generateScore(image1, image2):
        img1 = imageEncoder(image1)
        img2 = imageEncoder(image2)
        cos_scores = util.pytorch_cos_sim(img1, img2)
        score = round(float(cos_scores[0][0]) * 100, 2)
        return score

    return generateScore(img1, img2)


# function to compute similarity using histogram
def compute_histogram(img1, img2):
    # Ensure images are in BGR format
    if len(img1.shape) == 2:  # Grayscale image
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:  # Grayscale image
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Convert images to HSV color space
    hsv_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    # Compute the histogram for each image
    hist_img1 = cv2.calcHist([hsv_img1], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
    hist_img2 = cv2.calcHist([hsv_img2], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])

    # Normalize the histograms
    cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Compare the histograms using the correlation method
    similarity = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
    return similarity


# function to compute similarity using SSIM
def compute_ssim(img1, img2):
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))  # Resize img1 to match img2
    similarity, _ = ssim(img1, img2, full=True)
    return similarity


# function to compute similarity using ORB
def compute_orb(img1, img2, match_threshold):
    # Initialize the ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Initialize the matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort matches by distance (lower distance is better)
    matches = sorted(matches, key=lambda x: x.distance)

    # Calculate similarity based on the number of good matches
    similarity = len(matches)
    return similarity >= match_threshold

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True)

# Specify video source and output directory
video_path = input("Enter the video source file path: ")
output_dir = input("Enter the output directory to save frames: ")
os.makedirs(output_dir, exist_ok=True)

# Load reference images
first_image_path = 'first.jpg'
second_image_path = 'second.jpg'
first_image = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
second_image = cv2.imread(second_image_path, cv2.IMREAD_GRAYSCALE)

# Get the similarity threshold, confidence threshold, and half precision option from user
comparison_method = int(input("Choose comparison method: 1 for SSIM, 2 for ORB, 3 for Histogram, 4 for CLIP based: "))
if comparison_method == 1:
    similarity_threshold = float(input("Enter the similarity threshold for SSIM (e.g., 0.1 for lenient, 0.9 for strict): "))
elif comparison_method == 2:
    match_threshold = int(input("Enter the number of matches threshold for ORB: "))
elif comparison_method == 3:
    similarity_threshold = float(input("Enter the similarity threshold for Histogram (e.g., 0.1 for lenient, 0.9 for strict): "))
elif comparison_method == 4:
    similarity_threshold = float(input("Enter the similarity threshold for CLIP (e.g., 10 for lenient, 90 for strict): "))


conf_thresh = float(input("Enter the confidence threshold (e.g., 0.25): "))
use_half = input("Use half precision? (yes/no): ").lower() == 'yes'

# Open video file
cap = cv2.VideoCapture(video_path)
frame_count = 0

# Define the percentage to increase the bounding box size
bbox_expansion_factor = 0.2  # 20% larger

if use_half:
    model.half()  # Use half precision if specified

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Print the current frame number
    print(f"Processing frame {frame_count}...")

    # Process frame with YOLOv5 model
    results = model(frame)
    results = results.xyxy[0][results.xyxy[0][:, 4] >= conf_thresh]  # Apply confidence threshold

    # Extract results
    labels = results[:, -1].tolist()
    coords = results[:, :-1].tolist()
    classes = model.names

    person_count = 0
    for label, coord in zip(labels, coords):
        class_name = classes[int(label)]
        if class_name == 'person':
            person_count += 1

            # Draw bounding box
            x1, y1, x2, y2, conf = coord
            width = x2 - x1
            height = y2 - y1

            # Expand the bounding box
            x1_exp = max(0, x1 - bbox_expansion_factor * width)
            y1_exp = max(0, y1 - bbox_expansion_factor * height)
            x2_exp = min(frame.shape[1], x2 + bbox_expansion_factor * width)
            y2_exp = min(frame.shape[0], y2 + bbox_expansion_factor * height)

            # Convert coordinates to integers
            x1_px, y1_px, x2_px, y2_px = int(x1_exp), int(y1_exp), int(x2_exp), int(y2_exp)

            # Crop the region inside the bounding box
            cropped_frame = frame[y1_px:y2_px, x1_px:x2_px]
            cropped_frame_gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)

            # Compare the cropped frame with reference images
            if comparison_method == 1:
                first_similarity = compute_ssim(cropped_frame_gray, first_image)
                second_similarity = compute_ssim(cropped_frame_gray, second_image)
                if first_similarity >= 0.5:
                    frame_path = os.path.join(output_dir, f'frame{frame_count}_x1_{x1_px}_y1_{y1_px}_x2_{x2_px}_y2_{y2_px}_first.jpg')
                    cv2.imwrite(frame_path, cropped_frame)
                    print(f'Saved cropped frame {frame_count}_x1_{x1_px}_y1_{y1_px}_x2_{x2_px}_y2_{y2_px}_first with SSIM similarity {first_similarity:.2f} to {frame_path}')
                if second_similarity >= 0.3:
                    frame_path = os.path.join(output_dir, f'frame{frame_count}_x1_{x1_px}_y1_{y1_px}_x2_{x2_px}_y2_{y2_px}_second.jpg')
                    cv2.imwrite(frame_path, cropped_frame)
                    print(f'Saved cropped frame {frame_count}_x1_{x1_px}_y1_{y1_px}_x2_{x2_px}_y2_{y2_px}_second with SSIM similarity {second_similarity:.2f} to {frame_path}')
            elif comparison_method == 2:
                first_similarity = compute_orb(cropped_frame_gray, first_image, match_threshold)
                second_similarity = compute_orb(cropped_frame_gray, second_image, match_threshold)
                if first_similarity:
                    frame_path = os.path.join(output_dir, f'frame{frame_count}_x1_{x1_px}_y1_{y1_px}_x2_{x2_px}_y2_{y2_px}_first.jpg')
                    cv2.imwrite(frame_path, cropped_frame)
                    print(f'Saved cropped frame {frame_count}_x1_{x1_px}_y1_{y1_px}_x2_{x2_px}_y2_{y2_px}_first with enough matches to {frame_path}')
                if second_similarity:
                    frame_path = os.path.join(output_dir, f'frame{frame_count}_x1_{x1_px}_y1_{y1_px}_x2_{x2_px}_y2_{y2_px}_second.jpg')
                    cv2.imwrite(frame_path, cropped_frame)
                    print(f'Saved cropped frame {frame_count}_x1_{x1_px}_y1_{y1_px}_x2_{x2_px}_y2_{y2_px}_second with enough matches to {frame_path}')
            elif comparison_method == 3:
                first_similarity = compute_histogram(cropped_frame, first_image)
                second_similarity = compute_histogram(cropped_frame, second_image)
                if first_similarity >= similarity_threshold:
                    frame_path = os.path.join(output_dir, f'frame{frame_count}_x1_{x1_px}_y1_{y1_px}_x2_{x2_px}_y2_{y2_px}_first.jpg')
                    cv2.imwrite(frame_path, cropped_frame)
                    print(f'Saved cropped frame {frame_count}_x1_{x1_px}_y1_{y1_px}_x2_{x2_px}_y2_{y2_px}_first with Histogram similarity {first_similarity:.2f} to {frame_path}')
                if second_similarity >= similarity_threshold:
                    frame_path = os.path.join(output_dir, f'frame{frame_count}_x1_{x1_px}_y1_{y1_px}_x2_{x2_px}_y2_{y2_px}_second.jpg')
                    cv2.imwrite(frame_path, cropped_frame)
                    print(f'Saved cropped frame {frame_count}_x1_{x1_px}_y1_{y1_px}_x2_{x2_px}_y2_{y2_px}_second with Histogram similarity {second_similarity:.2f} to {frame_path}')
            elif comparison_method == 4:
                first_similarity = compute_clip(cropped_frame, first_image)
                second_similarity = compute_clip(cropped_frame, second_image)
                if first_similarity >= similarity_threshold:
                    frame_path = os.path.join(output_dir, f'frame{frame_count}_x1_{x1_px}_y1_{y1_px}_x2_{x2_px}_y2_{y2_px}_first.jpg')
                    cv2.imwrite(frame_path, cropped_frame)
                    print(f'Saved cropped frame {frame_count}_x1_{x1_px}_y1_{y1_px}_x2_{x2_px}_y2_{y2_px}_first with CLIP similarity {first_similarity:.2f} to {frame_path}')
                if second_similarity >= similarity_threshold:
                    frame_path = os.path.join(output_dir, f'frame{frame_count}_x1_{x1_px}_y1_{y1_px}_x2_{x2_px}_y2_{y2_px}_second.jpg')
                    cv2.imwrite(frame_path, cropped_frame)
                    print(f'Saved cropped frame {frame_count}_x1_{x1_px}_y1_{y1_px}_x2_{x2_px}_y2_{y2_px}_second with CLIP similarity {second_similarity:.2f} to {frame_path}')

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
