import cv2
import numpy as np
import os
import sys

def add_shot_noise(image):
    """
    Add shot noise (Poisson noise) to an image.

    Parameters:
    - image: Input image

    Returns:
    - noisy_image: Image with shot noise added
    """
    # Normalize the image to the range [0, 1]
    image = image / 255.0
    # Apply Poisson noise
    noisy_image = np.random.poisson(image * 255.0) / 255.0
    # Clip the values to the range [0, 1] and convert back to uint8
    noisy_image = np.clip(noisy_image * 255.0, 0, 255).astype('uint8')
    return noisy_image

def process_image(input_image_path):
    # Read the input image
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Unable to read image {input_image_path}")
        return

    # Add shot noise to the image
    noisy_image = add_shot_noise(image)

    # Generate the output image path based on the input image name
    base_name = os.path.basename(input_image_path)
    name, ext = os.path.splitext(base_name)
    output_image_path = f"{name}_shot{ext}"

    # Save the noisy image
    cv2.imwrite(output_image_path, noisy_image)

    # Display the original and noisy images
    cv2.imshow('Original Image', image)
    cv2.imshow('Noisy Image', noisy_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Noisy image saved as {output_image_path}")

def process_video(input_video_path):
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {input_video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Generate the output video path based on the input video name
    base_name = os.path.basename(input_video_path)
    name, ext = os.path.splitext(base_name)
    output_video_path = f"{name}_shot{ext}"

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Add shot noise to the frame
        noisy_frame = add_shot_noise(frame)

        # Write the frame to the output video
        out.write(noisy_frame)

    # Release everything if job is finished
    cap.release()
    out.release()

    print(f"Noisy video saved as {output_video_path}")

def main(input_path):
    # Check if the input path is an image or a video
    ext = os.path.splitext(input_path)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        process_image(input_path)
    elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
        process_video(input_path)
    else:
        print("Error: Unsupported file format")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python add_shot_noise.py <input_image_or_video_path>")
    else:
        input_path = sys.argv[1]
        main(input_path)