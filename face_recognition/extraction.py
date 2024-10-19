import os
import cv2
import torch
from facenet_pytorch import MTCNN
from PIL import Image

# Initialize MTCNN for face detection (using CPU or GPU if available)
if torch.cuda.is_available():
    print('CUDA')
else:
    print('ghanta')
# mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device='cpu')

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


# def extract_frames(video_path, output_dir, frame_interval=30, extract_faces=False):
#     """
#     Extract frames from a video and optionally extract faces from the frames.
    
#     Parameters:
#     - video_path: Path to the input video file
#     - output_dir: Directory to save the extracted frames and faces
#     - frame_interval: Save every nth frame
#     - extract_faces: Boolean flag to enable face extraction from frames
#     """
#     # Check if directory exists before trying to create it
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         print("Created output directory:", output_dir)
#     else:
#         print(f"Directory {output_dir} already exists.")

#     # Create a directory for extracted faces if face extraction is enabled
#     if extract_faces:
#         face_output_dir = os.path.join(output_dir, 'faces')
#         if not os.path.exists(face_output_dir):
#             os.makedirs(face_output_dir)
#             print(f"Created face output directory: {face_output_dir}")
#     else:
#         face_output_dir = None

#     vid = cv2.VideoCapture(video_path)

#     if not vid.isOpened():
#         print('Could not open video:', video_path)
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
#             print(f"Saved frame: {frame_filename}")
#             saved_frame_count += 1

#             # If face extraction is enabled, extract faces from the frame
#             if extract_faces:
#                 extract_faces_from_frame(frame, face_output_dir, saved_frame_count)

#         frame_count += 1

#     # Release the video capture object
#     vid.release()
#     print(f"Extracted {saved_frame_count} frames and saved to {output_dir}")

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
                try:
                    # Initialize face detector (MTCNN)
                    detector = MTCNN(keep_all=True, device='cpu')

                    # Detect faces and get bounding boxes
                    boxes, _ = detector.detect(frame)

                    if boxes is not None:
                        # Get frame dimensions
                        height, width, _ = frame.shape

                        for i, box in enumerate(boxes):
                            # Original bounding box coordinates
                            x1, y1, x2, y2 = box
                            box_width = x2 - x1
                            box_height = y2 - y1

                            # Expand the bounding box (enlarging by 20%)
                            scale_factor = 1.2
                            x1_new = max(0, x1 - (box_width * (scale_factor - 1) / 2))
                            y1_new = max(0, y1 - (box_height * (scale_factor - 1) / 2))
                            x2_new = min(width, x2 + (box_width * (scale_factor - 1) / 2))
                            y2_new = min(height, y2 + (box_height * (scale_factor - 1) / 2))

                            # Crop the enlarged face region
                            face = frame[int(y1_new):int(y2_new), int(x1_new):int(x2_new)]

                            # Save the cropped face
                            face_filename = os.path.join(face_output_dir, f"face_{saved_frame_count:05d}_{i}.jpg")
                            cv2.imwrite(face_filename, face)
                            print(f"Saved enlarged face: {face_filename}")
                    else:
                        print(f"No faces detected in frame {saved_frame_count}")

                except Exception as e:
                    print(f"Error processing frame {saved_frame_count} for face extraction: {e}")

        frame_count += 1

    # Release the video capture object
    vid.release()
    print(f"Extracted {saved_frame_count} frames and saved to {output_dir}")

