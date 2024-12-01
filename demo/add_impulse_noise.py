import cv2
import numpy as np
import os
import sys

def add_impulse_noise(image, prob=0.05):
    """
    Add impulse noise (salt-and-pepper noise) to an image.

    Parameters:
    - image: Input image
    - prob: Probability of noise

    Returns:
    - noisy_image: Image with impulse noise added
    """
    noisy_image = np.copy(image)
    # Salt noise
    num_salt = np.ceil(prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 255

    # Pepper noise
    num_pepper = np.ceil(prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1], :] = 0

    return noisy_image

def process_image(input_image_path, prob):
    # Read the input image
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Unable to read image {input_image_path}")
        return

    # Add impulse noise to the image
    noisy_image = add_impulse_noise(image, prob=prob)

    # Generate the output image path based on the input image name
    base_name = os.path.basename(input_image_path)
    name, ext = os.path.splitext(base_name)
    output_image_path = f"{name}_impulse{ext}"

    # Save the noisy image
    cv2.imwrite(output_image_path, noisy_image)

    # Display the original and noisy images
    cv2.imshow('Original Image', image)
    cv2.imshow('Noisy Image', noisy_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Noisy image saved as {output_image_path}")

def process_video(input_video_path, prob):
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
    output_video_path = f"{name}_impulse{ext}"

    # Create a VideoWriter object
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Add impulse noise to the frame
        noisy_frame = add_impulse_noise(frame, prob=prob)

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
    prob = 0.01
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            process_image(input_path, prob)
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            process_video(input_path, prob)
        else:
            print(f"Unsupported file format: {ext}")
    else:
        print(f"Error: {input_path} is not a valid file")

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python add_impulse_noise.py <input_image_or_video_path> [probability]")
    else:
        input_path = sys.argv[1]
        prob = float(sys.argv[2]) if len(sys.argv) == 3 else 0.05  # Default probability is 0.05
        main(input_path)