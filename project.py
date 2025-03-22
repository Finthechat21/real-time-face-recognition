try:
    import cv2
    import face_recognition
    import numpy as np
    import threading
except ImportError as e:
    print(f"Error: {e}. Please install the required libraries.")
    exit()

known_faces = []
known_names = []

known_images = ["moetaz.jpg"]
names = ["moetaz"]

if len(known_images) != len(names):
    raise ValueError("The lengths of known_images and names must be equal.")

for img_path, name in zip(known_images, names):
    try:
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_faces.append(encodings[0])
            known_names.append(name)
    except Exception as e:
        print(f"⚠️ Error processing {img_path}: {e}")

video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not access the webcam. Please check if the webcam is connected and not being used by another application.")
    exit()

fps = video_capture.get(cv2.CAP_PROP_FPS)
wait_time = int(1000 / fps) if fps > 1 else 1

GREEN_COLOR = (0, 255, 0)
GREEN_TEXT_COLOR = (0, 255, 0)

def display_frame():
    while True:
        if display_frame.frame is not None:
            cv2.imshow('Real-Time Face Recognition', display_frame.frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

display_frame.frame = None
display_thread = threading.Thread(target=display_frame)
display_thread.start()

try:
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Failed to read frame from webcam. Exiting...")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_faces, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), GREEN_COLOR, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN_TEXT_COLOR, 2)

        display_frame.frame = frame

        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
finally:
    video_capture.release()
    cv2.destroyAllWindows()
