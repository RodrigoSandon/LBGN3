import cv2
import numpy as np
from edge_connect import EdgeConnect

# Initialize the EdgeConnect model
edgeconnect = EdgeConnect()
edgeconnect.load_model("models/Places2")

# Open input video file
cap = cv2.VideoCapture('input_video.mp4')

# Create background subtractor object
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Load reference background image and resize it to match video frames
bg_img = cv2.imread('reference_background.jpg')
bg_img = cv2.resize(bg_img, (int(cap.get(3)), int(cap.get(4))))

# Open output video file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Set maximum number of frames to process
n = 100  # Replace with the desired number of frames to process

# Process each frame in the input video
frame_count = 0
while frame_count < n:
    # Read frame from input video
    ret, frame = cap.read()
    
    if not ret:
        break

    # Apply background subtraction
    mask = bg_subtractor.apply(frame)

    # Apply thresholding to create binary mask
    threshold = 100
    _, binary_mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)

    # Apply morphology operations to remove noise and fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Invert binary mask to keep non-moving pixels
    binary_mask_inv = cv2.bitwise_not(binary_mask)

    # Apply inverted binary mask to original frame to keep non-moving pixels
    result = cv2.bitwise_and(frame, frame, mask=binary_mask_inv)

    # Use the EdgeConnect model to inpaint the masked areas
    inpainted_frame = edgeconnect.inpaint(frame, binary_mask)

    # Merge the inpainted frame with the original frame using the inverted binary mask
    result = cv2.addWeighted(result, 1, inpainted_frame, 0, 0, binary_mask_inv)

    # Write processed frame to output video
    out.write(result)
    
    frame_count += 1

# Release video capture and writer objects
cap.release()
out.release()
