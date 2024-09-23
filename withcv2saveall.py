import torch
import cv2
import os
from skimage.metrics import structural_similarity as ssim

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True)

# Specify video source and output directory
video_path = input("Enter the video source file path: ")
output_dir = input("Enter the output directory to save frames: ")
os.makedirs(output_dir, exist_ok=True)

# Load reference images
first_image_path = 'first.jpg'
second_image_path = 'second.jpg'
first_image = cv2.imread(first_image_path)
second_image = cv2.imread(second_image_path)

# Get the similarity threshold and confidence threshold from user
similarity_threshold = float(input("Enter the similarity threshold (e.g., 0.1 for lenient, 0.9 for strict): "))
conf_threshold = float(input("Enter the confidence threshold (e.g., 0.25): "))

# Function to compute similarity between two images using SSIM
def compute_similarity(img1, img2):
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))  # Resize img1 to match img2
    similarity = ssim(img1, img2)
    return similarity

# Function to check if the dimensions are within 10% of each other
def are_sizes_within_10_percent(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    return (0.9 * h1 <= h2 <= 1.1 * h1) and (0.9 * w1 <= w2 <= 1.1 * w1)

# Open video file
cap = cv2.VideoCapture(video_path)
frame_count = 0

# Define the percentage to increase the bounding box size
bbox_expansion_factor = 0.2  # 20% larger

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Print the current frame number
    print(f"Processing frame {frame_count}...")

    # Process frame with YOLOv5 model
    results = model(frame)  # Adjust image size
    results = results.xyxy[0][results.xyxy[0][:, 4] >= conf_threshold]  # Apply confidence threshold

    # Extract results
    labels = results[:, -1].tolist()
    coords = results[:, :-1].tolist()
    classes = model.names

    object_count = 0
    for label, coord in zip(labels, coords):
        class_name = classes[int(label)]
        object_count += 1

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
        cropped_frame_gray = cropped_frame

        # Compare the cropped frame with reference images if sizes are within 10%
        if are_sizes_within_10_percent(cropped_frame_gray, first_image):
            first_similarity = compute_similarity(cropped_frame_gray, first_image)
            if first_similarity >= 0.5:
                frame_path = os.path.join(output_dir, f'frame{frame_count}_x1_{x1_px}_y1_{y1_px}_x2_{x2_px}_y2_{y2_px}_{class_name}_first.jpg')
                cv2.imwrite(frame_path, cropped_frame)
                print(f'Saved cropped frame {frame_count}_x1_{x1_px}_y1_{y1_px}_x2_{x2_px}_y2_{y2_px}_{class_name}_first with similarity {first_similarity:.2f} to {frame_path}')

        if are_sizes_within_10_percent(cropped_frame_gray, second_image):
            second_similarity = compute_similarity(cropped_frame_gray, second_image)
            if second_similarity >= 0.2:
                frame_path = os.path.join(output_dir, f'frame{frame_count}_x1_{x1_px}_y1_{y1_px}_x2_{x2_px}_y2_{y2_px}_{class_name}_second.jpg')
                cv2.imwrite(frame_path, cropped_frame)
                print(f'Saved cropped frame {frame_count}_x1_{x1_px}_y1_{y1_px}_x2_{x2_px}_y2_{y2_px}_{class_name}_second with similarity {second_similarity:.2f} to {frame_path}')

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
