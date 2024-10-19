from extraction import extract_frames
from model import run_codeformer_again
from db import redis_init, vector_query, get_images, filter_documents_by_distance, save_images_from_ids
import os
# import zipfile36 as zipfile
import zipfile

# DEFAULT VALUES
VID_EXTRACTION_RES = './VidExtractionFrames'
FRAME_INTERVAL = 30
MODEL_RESULTS_PATH = './ModelRes'
MODEL_WEIGHT=0.5
DATASET_DIR = "Celebrity Faces Dataset"


# def workflow(video_path):
#   extract_frames(video_path, output_dir=VID_EXTRACTION_RES, frame_interval=FRAME_INTERVAL, extract_faces=True)
#   run_codeformer_again(input_path=VID_EXTRACTION_RES, output_path=MODEL_RESULTS_PATH, weight=MODEL_WEIGHT)

# def face_recognition(img_path):
#   run_codeformer_again(input_path=img_path, output_path=MODEL_RESULTS_PATH, weight=MODEL_WEIGHT)
#   r = redis_init()
#   restored_image_name = os.path.basename(img_path)  # Extract just the image name
#   restored_image_path = os.path.join(MODEL_RESULTS_PATH, "restored_faces", restored_image_name)

#     # Check if the restored image exists
#   if not os.path.exists(restored_image_path):
#         print(f"Restored image not found: {restored_image_path}")
#         return

#     # Step 3: Perform vector search on the restored image
#   print(f"Running vector search on restored image: {restored_image_path}")
#   matching_documents = vector_query(img_path=restored_image_path)

#   get_images(dataset_dir=DATASET_DIR, output_dir=MATCHES, results=matching_documents)



def process_video_workflow(video_path, output_dir, frame_interval=30):
    print('Video processing')
    """
    Workflow to extract frames and faces from video, restore faces, and return a zip file of restored faces.
    """
    extract_frames(video_path, output_dir, frame_interval, extract_faces=True)

    face_dir = os.path.join(output_dir, 'faces')
    run_codeformer_again(input_path=face_dir, output_path=output_dir)

    restored_faces_dir = os.path.join(output_dir, 'final_results')
    zip_filename = os.path.join(output_dir, 'restored_faces.zip')
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for root, dirs, files in os.walk(restored_faces_dir):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), restored_faces_dir))

    return zip_filename

def image_matching_workflow(img_path, base_directory, output_directory):
    """
    Workflow to restore an image, run vector query, and return matched images.
    """
    print(
        'image recognition'
    )
    r = redis_init()

    restored_faces_dir = os.path.join(output_directory, 'final_results')
    run_codeformer_again(input_path=img_path, output_path=output_directory)

    restored_img_path = os.path.join(restored_faces_dir, os.path.basename(img_path))

    matching_docs = vector_query(img_path=restored_img_path, r=r)
    print(f'MATCHES: {matching_docs}')

    filtered_docs = filter_documents_by_distance(matching_docs)

    matched_faces_dir = os.path.join(output_directory, 'matched_faces')
    save_images_from_ids(filtered_docs, output_directory=matched_faces_dir, base_directory=base_directory)

    matched_image_paths = [os.path.join(matched_faces_dir, doc['id']) for doc in filtered_docs]
    return matched_image_paths

# res = image_matching_workflow(img_path="hughJackman.png", base_directory=DATASET_DIR, output_directory='./MATCH')
# print(res)