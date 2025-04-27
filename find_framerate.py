import cv2


# path to the video file
video_path = 'sample.mp4'

# open video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Framerate (fps): {fps}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolution: {frame_width} x {frame_height}")

# release the video capture object
cap.release()

########################################################################################

# import cv2
#
# # path to your video file
# video_path = 'sample.mp4'
#
# # open the video file
# cap = cv2.VideoCapture(video_path)
#
# # check if the video opened successfully
# if not cap.isOpened():
#     print("Error: Could not open video.")
# else:
#     # get the frame rate (fps)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#
#     # get the frame width and height
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     # calculate half width and height
#     half_width = frame_width // 2
#     half_height = frame_height // 2
#
#     # define the codec and create VideoWriter objects for each part
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video format
#
#     out_top_left = cv2.VideoWriter('top_left.mp4', fourcc, fps, (half_width, half_height))
#     out_top_right = cv2.VideoWriter('top_right.mp4', fourcc, fps, (half_width, half_height))
#     out_bottom_left = cv2.VideoWriter('bottom_left.mp4', fourcc, fps, (half_width, half_height))
#     out_bottom_right = cv2.VideoWriter('bottom_right.mp4', fourcc, fps, (half_width, half_height))
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # split the frame into four parts
#         top_left = frame[:half_height, :half_width]
#         top_right = frame[:half_height, half_width:]
#         bottom_left = frame[half_height:, :half_width]
#         bottom_right = frame[half_height:, half_width:]
#
#         # write each part to the corresponding video file
#         out_top_left.write(top_left)
#         out_top_right.write(top_right)
#         out_bottom_left.write(bottom_left)
#         out_bottom_right.write(bottom_right)
#
#     # release the video capture and writer objects
#     cap.release()
#     out_top_left.release()
#     out_top_right.release()
#     out_bottom_left.release()
#     out_bottom_right.release()
#     cv2.destroyAllWindows()
#
#     print("Videos saved successfully.")

########################################################################################

# find structural similiarity
import cv2
from skimage.metrics import structural_similarity as ssim

# load the images
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')

# Convert images to grayscale
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Ensure the images are the same size
image1_gray = cv2.resize(image1_gray, (image2_gray.shape[1], image2_gray.shape[0]))

# Compute SSIM between the two images
ssim_index, ssim_map = ssim(image1_gray, image2_gray, full=True)
print(f"SSIM: {ssim_index}")

# Optionally, show the SSIM image
import matplotlib.pyplot as plt
plt.imshow(ssim_map, cmap='gray')
plt.title(f"SSIM Map - Index: {ssim_index:.5f}")
plt.colorbar()
plt.show()

########################################################################################

# # feature-based methods
# import cv2

# # load the images
# image1_path = 'image1.jpg'
# image2_path = 'image2.jpg'
# img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

# # initialize the ORB detector
# orb = cv2.ORB_create()

# # detect keypoints and descriptors
# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2, None)

# # initialize the matcher
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# # match descriptors
# matches = bf.match(des1, des2)

# # sort matches by distance (lower distance is better)
# matches = sorted(matches, key=lambda x: x.distance)

# # draw top N matches (optional)
# N = 10
# img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:N], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# # show the matches
# cv2.imshow('Matches', img_matches)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # calculate similarity based on the number of good matches
# similarity = len(matches)
# print(f"Number of good matches: {similarity}")

########################################################################################

# # using histogram based method
# import cv2
# # load images
# image1 = cv2.imread('image1.jpg')
# image2 = cv2.imread('image2.jpg')
# hist_img1 = cv2.calcHist([image1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
# hist_img1[255, 255, 255] = 0 # gnore all white pixels
# cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
# hist_img2 = cv2.calcHist([image2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
# hist_img2[255, 255, 255] = 0  # ignore all white pixels
# cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
# # find the metric value
# metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
# print(f"Similarity Score: ", round(metric_val, 2))
# # similarity score: 0.94

########################################################################################

# import cv2
# import torch
# import open_clip
# from PIL import Image
# from sentence_transformers import util

# # load images
# image1_path = 'image1.jpg'
# image2_path = 'image2.jpg'

# # image processing model
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
# model.to(device)

# def imageEncoder(img):
#     img1 = Image.fromarray(img).convert('RGB')
#     img1 = preprocess(img1).unsqueeze(0).to(device)
#     img1 = model.encode_image(img1)
#     return img1

# def generateScore(image1_path, image2_path):
#     test_img = cv2.imread(image1_path, cv2.IMREAD_UNCHANGED)
#     data_img = cv2.imread(image2_path, cv2.IMREAD_UNCHANGED)
#     img1 = imageEncoder(test_img)
#     img2 = imageEncoder(data_img)
#     cos_scores = util.pytorch_cos_sim(img1, img2)
#     score = round(float(cos_scores[0][0]) * 100, 2)
#     return score

# print(f"Similarity Score: {generateScore(image1_path, image2_path)}")
