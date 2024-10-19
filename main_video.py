import cv2
import os
import glob
import numpy as np
import face_recognition

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25  # Resize frame for faster processing

    def load_encoding_images(self, images_path):
        """
        Load encoding images from the specified path
        :param images_path: Path to the folder containing images
        """
        # Load Images
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        print(f"{len(images_path)} encoding images found.")

        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            filename, _ = os.path.splitext(basename)

            # Get encoding
            img_encoding = face_recognition.face_encodings(rgb_img)
            if img_encoding:  # Check if encoding was found
                self.known_face_encodings.append(img_encoding[0])
                self.known_face_names.append(filename)
            else:
                print(f"No face found in the image: {basename}")

        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        """
        Detect known faces in the given frame
        :param frame: The current frame from the video
        :return: Detected face locations and names
        """
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Check if there are any matches
            if matches:
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations) / self.frame_resizing
        return face_locations.astype(int), face_names


# Main code to capture video and recognize faces
if __name__ == "__main__":
    sfr = SimpleFacerec()
    sfr.load_encoding_images("images")  # Specify the path to your image folder

    video_capture = cv2.VideoCapture(0)  # Use 0 for webcam or specify video file path

    frame_skip = 5  # Process every 5th frame
    frame_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture video")
            break
        
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip processing this frame

        # Resize the frame
        frame = cv2.resize(frame, (640, 480))

        # Detect known faces
        try:
            face_locations, face_names = sfr.detect_known_faces(frame)
        except Exception as e:
            print(f"Error in detecting faces: {e}")
            continue

        # Draw the results on the frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
