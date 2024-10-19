import os
import torch
import cv2
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed 
from facenet_pytorch import MTCNN
from PIL import Image
import time as time


if torch.cuda.is_available():
    print("CUDA is available.")
    print(torch.cuda.get_device_name())
else:
    print("CUDA is not available. Please check your PyTorch installation.")

input_vid = "video\\circus.mp4"
output_dir= ".\\result"

mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

def extract_faces_from_frame(frame, output_dir, frame_number):
    """
    Detects and extracts faces from the given frame and saves them to output_dir.
    """
    try:
        # Convert OpenCV BGR image to PIL RGB image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Detect faces
        boxes, _ = mtcnn.detect(pil_image)
        if boxes is not None:
            for i, box in enumerate(boxes):
                # Adjust bounding box to ensure full face is captured
                x1, y1, x2, y2 = map(int, box)
                face = frame[y1:y2, x1:x2]

                # Save the extracted face
                face_filename = os.path.join(output_dir, f"frame_{frame_number:05d}_face_{i}.jpg")
                cv2.imwrite(face_filename, face)
                print(f"Saved face: {face_filename}")
    except Exception as e:
        print(f"Error processing frame {frame_number} for face extraction: {e}")


def extract_frames(video_path, output_dir, frame_interval=30, extract_faces=False):
    """
    Extract frames from a video and optionally extract faces from the frames.
    
    Parameters:
    - video_path: Path to the input video file
    - output_dir: Directory to save the extracted frames and faces
    - frame_interval: Save every nth frame
    - extract_faces: Boolean flag to enable face extraction from frames
    """
    # Check if directory exists before trying to create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Created output directory:", output_dir)
    else:
        print(f"Directory {output_dir} already exists.")

    # Create a directory for extracted faces if face extraction is enabled
    if extract_faces:
        face_output_dir = os.path.join(output_dir, 'faces')
        if not os.path.exists(face_output_dir):
            os.makedirs(face_output_dir)
            print(f"Created face output directory: {face_output_dir}")
    else:
        face_output_dir = None

    vid = cv2.VideoCapture(video_path)

    if not vid.isOpened():
        print('Could not open video:', video_path)
        return

    frame_count = 0
    saved_frame_count = 0

    while True:
        # Read the next frame from the video
        success, frame = vid.read()

        # If reading a frame was not successful, we are at the end of the video
        if not success:
            break

        # Save every `frame_interval`-th frame
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame: {frame_filename}")
            saved_frame_count += 1

            # If face extraction is enabled, extract faces from the frame
            if extract_faces:
                extract_faces_from_frame(frame, face_output_dir, saved_frame_count)

        frame_count += 1

    # Release the video capture object
    vid.release()
    print(f"Extracted {saved_frame_count} frames and saved to {output_dir}")



# def extract_frames(video_path, output_dir, frame_interval=30):
#     # Check if directory exists before trying to create it
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         print("Created output dir")
#     else:
#         print(f"Directory {output_dir} already exists.")

#     vid = cv2.VideoCapture(video_path)

#     if not vid.isOpened():
#         print('Could not open: ', video_path)
#         return 
#     frame_count = 0
#     saved_frame_count = 0

#     while True:
#         # Read the next frame from the video
#         success, frame = vid.read()

#         # If reading a frame was not successful, we are at the end of the video
#         if not success:
#             break

#         # Save every `frame_interval`-th frame
#         if frame_count % frame_interval == 0:
#             frame_filename = os.path.join(output_dir, f"frame_{saved_frame_count:05d}.jpg")
#             cv2.imwrite(frame_filename, frame)
#             saved_frame_count += 1

#         frame_count += 1

#     # Release the video capture object
#     vid.release()
#     print(f"Extracted {saved_frame_count} frames and saved to {output_dir}")

def run_codeformer_multithreaded(input_dir, output_dir, num_threads=None):
    '''
    Run codeformer in multithreaded mode
    '''
    # Convert input and output directories to absolute paths
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    
    # Get list of images in the input directory
    input_images = [img for img in os.listdir(input_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]

    # Dynamically determine the number of threads based on CPU cores if not provided
    if num_threads is None:
        num_threads = os.cpu_count()  # Set to the number of available CPU cores

    print(f"Using {num_threads} threads for processing...")

    # Use a ThreadPoolExecutor for multithreaded execution
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(run_codeformer, img, input_dir, output_dir) for img in input_images]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing image: {e}")


def run_codeformer(input_path, output_path, weight=0.5):
    """
    Run the CodeFormer inference script with the given input and output paths using GPU if available.

    :param input_path: Absolute path of the input image
    :param output_path: Absolute path of the output directory
    :param weight: Weight parameter for the CodeFormer script
    """
    # Convert paths to absolute paths
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)

    # Adjust the Python executable to point to the virtual environment's Python if necessary
    python_executable = os.path.join(os.environ.get('VIRTUAL_ENV', ''), 'Scripts', 'python.exe')
    
    if not os.path.exists(python_executable):
        # If no virtual environment, fallback to system Python
        python_executable = 'python'
    
    # Check if GPU is available and set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Construct the command
    command = [
        python_executable,  # Use the virtual environment's Python
        "CodeFormer\\inference_codeformer.py",
        "-w", str(weight),
        "--input_path", input_path,
        "--output_path", output_path,
        "--device", device  # Specify device (GPU or CPU)
    ]
    
    print(f"Running command: {' '.join(command)}")

    # Execute the command
    subprocess.run(command)
# Adjust the output directory path to avoid escape character issues
video = 'video\\circus.mp4'
output_dir = '.\\result\\gen'

def process_images_in_parallel(image_paths, output_directory, weight=0.5, max_workers=4):
    """
    Process multiple images in parallel using multithreading.
    
    :param image_paths: List of input image paths to be processed
    :param output_directory: Output directory where the results will be saved
    :param weight: Weight parameter for the CodeFormer script
    :param max_workers: Maximum number of threads to use for parallel processing
    """
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Use a ThreadPoolExecutor to process images in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for image_path in image_paths:
            output_path = os.path.join(output_directory, os.path.basename(image_path))
            futures.append(executor.submit(run_codeformer, image_path, output_path, weight))
        
        # Wait for all threads to complete
        for future in futures:
            future.result()

def run_codeformer_again (input_path, output_path, weight=0.5):
    """
    Run the CodeFormer inference script with the given input and output paths.
    
    :param input_path: Absolute path of the input image
    :param output_path: Absolute path of the output directory
    :param weight: Weight parameter for the CodeFormer script
    """
    # Convert paths to absolute paths
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)

    # Adjust the python executable to point to the virtual environment's Python if necessary
    python_executable = os.path.join(os.environ.get('VIRTUAL_ENV', ''), 'Scripts', 'python.exe')
    
    if not os.path.exists(python_executable):
        # If no virtual environment, fallback to system Python
        python_executable = 'python'
    
    # Construct the command
    command = [
        python_executable,  # Use the virtual environment's Python
        "CodeFormer\\inference_codeformer.py",
        "-w", str(weight),
        "--input_path", input_path,
        "--output_path", output_path
    ]
    
    print(f"Running command: {' '.join(command)}")
    
    # Execute the command
    subprocess.run(command)

# Call the function to extract frames
# extract_frames(video, output_dir=output_dir, frame_interval=10, extract_faces=True)

# start = time.time()
# # run_codeformer_multithreaded(output_dir, '.\\CF_results')
# run_codeformer_again(input_path='.\\result\\faces', output_path='.\\CF_results')
# end = time.time()
# print(f'Completed in: {end-start}')