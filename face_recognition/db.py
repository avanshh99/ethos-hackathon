import redis
from PIL import Image
import os
from redis.commands.search.field import VectorField, TagField
from redis.commands.search.query import Query
from deepface import DeepFace
import numpy as np

r = redis.Redis(
  host='redis-17799.c253.us-central1-1.gce.redns.redis-cloud.com',
  port=17799,
  password='uAmOPKyr5wZQ0DskgcWaxzJWW3Jatger',
  ssl = False,
  )

print(r.ping())

def redis_init():
    r = redis.Redis(
        host='redis-17799.c253.us-central1-1.gce.redns.redis-cloud.com',
        port=17799,
        password='uAmOPKyr5wZQ0DskgcWaxzJWW3Jatger',
        ssl = False,
    )
    return r

def vector_query(img_path, r, num_nearest = 4):
    target_embedding = DeepFace.represent(
        img_path=img_path,
        model_name="Facenet",
        detector_backend="mtcnn"
    )[0]["embedding"]

    query_vector = np.array(target_embedding).astype(np.float32).tobytes()
    k = num_nearest

    base_query = f"*=>[KNN {k} @embedding $query_vector AS distance]"
    query = Query(base_query).return_fields("distance").sort_by("distance").dialect(2)
    results = r.ft().search(query, query_params={"query_vector": query_vector})
    return results.docs


base_directory = 'Celebrity Faces Dataset'  # Change this to your local image directory
output_directory = os.path.join(base_directory, 'matched_faces')

def get_images(dataset_dir, output_dir, results, threshold=None):
    filter_docs = filter_documents_by_distance(results, max_distance=threshold)

    save_images_from_ids(filter_docs, base_directory=dataset_dir, output_directory=output_dir)

# Function to filter documents by distance <= 50
def filter_documents_by_distance(docs, max_distance=50):
    return [doc for doc in docs if float(doc['distance']) <= max_distance]

# Function to find and save images based on IDs
def save_images_from_ids(result_docs, output_directory, base_directory):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: {output_directory}")

    for doc in result_docs:
        image_id = doc['id']
        image_found = False  # Flag to check if the image is found

        # Traverse through the celebrity folders to find the image
        for celebrity_folder in os.listdir(base_directory):
            celebrity_path = os.path.join(base_directory, celebrity_folder)

            # Ensure we're working with a directory
            if os.path.isdir(celebrity_path):
                image_path = os.path.join(celebrity_path, image_id)

                # Check if the image exists in the current celebrity folder
                if os.path.exists(image_path):
                    # Open the image
                    img = Image.open(image_path)

                    # Save the image in the new directory
                    output_path = os.path.join(output_directory, image_id)
                    img.save(output_path)
                    print(f"Saved image: {output_path}")
                    image_found = True
                    break  # Exit the loop once the image is found

        if not image_found:
            print(f"Image {image_id} not found in any celebrity folder.")

# Filter documents based on distance <= 50
# filtered_docs = filter_documents_by_distance(results.docs, max_distance=50)

# Call the function to save images for the filtered documents
# save_images_from_ids(filtered_docs)
