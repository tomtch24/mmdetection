import cv2
import numpy as np
import os
import sys

def add_speckle_noise(image):
    """
    Add speckle noise to an image.

    Parameters:
    - image: Input image

    Returns:
    - noisy_image: Image with speckle noise added
    """
    noise = np.random.randn(*image.shape) * 0.1
    noisy_image = image + image * noise
    noisy_image = np.clip(noisy_image, 0, 255).astype('uint8')
    return noisy_image

def process_image(input_image_path):
    # Read the input image
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Unable to read image {input_image_path}")
        return

    # Add speckle noise to the image
    noisy_image = add_speckle_noise(image)

    # Generate the output image path based on the input image name
    base_name = os.path.basename(input_image_path)
    name, ext = os.path.splitext(base_name)
    output_image_path = f"{name}_speckle{ext}"

    # Save the noisy image
    cv2.imwrite(output_image_path, noisy_image)

    # Display the original and noisy images
    cv2.imshow('Original Image', image)
    cv2.imshow('Noisy Image', noisy_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Noisy image saved as {output_image_path}")

def process_video(input_video_path):
    # Read the input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Unable to read video {input_video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Generate the output video path based on the input video name
    base_name = os.path.basename(input_video_path)
    name, ext = os.path.splitext(base_name)
    output_video_path = f"{name}_speckle{ext}"

    # Create a VideoWriter object
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Add speckle noise to the frame
        noisy_frame = add_speckle_noise(frame)

        # Write the noisy frame to the output video
        out.write(noisy_frame)

        # Display the original and noisy frames
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Noisy Frame', noisy_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Noisy video saved as {output_video_path}")

def main(input_path):
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            process_image(input_path)
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            process_video(input_path)
        else:
            print(f"Unsupported file format: {ext}")
    else:
        print(f"Error: {input_path} is not a valid file")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python add_speckle_noise.py <input_image_or_video_path>")
    else:
        input_path = sys.argv[1]
        main(input_path)