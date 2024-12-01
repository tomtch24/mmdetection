import cv2
import os
import sys

def apply_gaussian_blur(image, kernel_size=10):
    """
    Apply Gaussian blur to an image.

    Parameters:
    - image: Input image
    - kernel_size: Size of the Gaussian blur kernel

    Returns:
    - blurred_image: Image with Gaussian blur applied
    """
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image

def process_image(input_image_path):
    # Read the input image
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Unable to read image {input_image_path}")
        return

    # Apply Gaussian blur to the image
    blurred_image = apply_gaussian_blur(image)

    # Generate the output image path based on the input image name
    base_name = os.path.basename(input_image_path)
    name, ext = os.path.splitext(base_name)
    output_image_path = f"{name}_gaussian_blur{ext}"

    # Save the blurred image
    cv2.imwrite(output_image_path, blurred_image)

    # Display the original and blurred images
    cv2.imshow('Original Image', image)
    cv2.imshow('Blurred Image', blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Blurred image saved as {output_image_path}")

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
    output_video_path = f"{name}_gaussian_blur{ext}"

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply Gaussian blur to the frame
        blurred_frame = apply_gaussian_blur(frame)

        # Write the frame to the output video
        out.write(blurred_frame)

    # Release everything if job is finished
    cap.release()
    out.release()

    print(f"Blurred video saved as {output_video_path}")

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
        print("Usage: python apply_gaussian_blur.py <input_image_or_video_path>")
    else:
        input_path = sys.argv[1]
        main(input_path)