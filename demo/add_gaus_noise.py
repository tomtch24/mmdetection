import cv2
import numpy as np
import os
import sys

def add_gaussian_noise(image, mean=0, var=0.1):
    """
    Add Gaussian noise to an image.

    Parameters:
    - image: Input image
    - mean: Mean of the Gaussian noise
    - var: Variance of the Gaussian noise

    Returns:
    - noisy_image: Image with Gaussian noise added
    """
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, image.shape).astype('uint8')
    noisy_image = cv2.add(image, gaussian)
    return noisy_image

def process_image(input_image_path, mean, var):
    # Read the input image
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Unable to read image {input_image_path}")
        return

    # Add Gaussian noise to the image
    noisy_image = add_gaussian_noise(image, mean=mean, var=var)

    # Generate the output image path based on the input image name
    base_name = os.path.basename(input_image_path)
    name, ext = os.path.splitext(base_name)
    output_image_path = f"{name}_gaus{ext}"

    # Save the noisy image
    cv2.imwrite(output_image_path, noisy_image)

    # Display the original and noisy images
    cv2.imshow('Original Image', image)
    cv2.imshow('Noisy Image', noisy_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Noisy image saved as {output_image_path}")

def process_video(input_video_path, mean, var):
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
    output_video_path = f"{name}_gaus{ext}"

    # Create a VideoWriter object
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Add Gaussian noise to the frame
        noisy_frame = add_gaussian_noise(frame, mean=mean, var=var)

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

def main(input_path, mean, var):
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            process_image(input_path, mean, var)
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            process_video(input_path, mean, var)
        else:
            print(f"Unsupported file format: {ext}")
    else:
        print(f"Error: {input_path} is not a valid file")

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python add_gaussian_noise.py <input_image_or_video_path> [mean] [variance]")
    else:
        input_path = sys.argv[1]
        mean = float(sys.argv[2]) if len(sys.argv) > 2 else 0  # Default mean is 0
        var = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1  # Default variance is 0.1
        main(input_path, mean, var)