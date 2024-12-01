import subprocess

def run_image_demo(config_file, checkpoint_file, image_files):
    for image_file in image_files:
        # Construct the command to run image_demo.py for each image
        command = [
            'python ',  # Path to the Python interpreter
            'demo/image_demo.py ',  # Path to the image_demo.py script
            image_file,
            config_file,'--weights',
            checkpoint_file           
        ]

        # Run the command
        subprocess.run(command)

if __name__ == "__main__":
    # Specify the input items directly in the script
    config_file = 'configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'
    checkpoint_file = 'checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
    image_files = [
        'demo/large_image.jpg',
        'demo/large_image_gaus.jpg',
        'demo/large_image_impulse.jpg',
        'demo/large_image_speckle.jpg',
        'demo/large_image_shot.jpg',
        'demo/large_image_gaussian_blur.jpg',
        'demo/large_image_glass_blur.jpg',
        'demo/large_image_motion_blur.jpg',
        'demo/large_image_defocus_blur.jpg',
        # Add more image paths as needed
    ]

    run_image_demo(config_file, checkpoint_file, image_files)