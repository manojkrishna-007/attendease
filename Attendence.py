import cv2
import os
import numpy as np
import csv
import json
import time
from datetime import datetime

# Initialize the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Initialize the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Fixed size for all faces to prevent matrix size mismatch errors
FACE_SIZE = (200, 200)

# Initialize the list of names mapping globally
names = {}

def train_model(base_path):
    """
    Trains the model using images from a given directory.
    Groups images accurately by subdirectories to fix labeling bugs.
    """
    images = []
    labels = []
    label_map = {}  # Map from person names to labels
    label_counter = 0

    # Walk through the base_path to get all the images
    for person_name in os.listdir(base_path):
        person_dir = os.path.join(base_path, person_name)
        
        # Skip the trained model outputs or single files if they exist in the directory
        if person_name in ["face_recognizer.yml", "labels_map.json"] or not os.path.isdir(person_dir):
            continue
            
        print(f"Processing images for: {person_name}")
        
        # Assign ONE unique integer label per person directory
        label_map[label_counter] = person_name
        
        for image_name in os.listdir(person_dir):
            if image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(person_dir, image_name)
                img = cv2.imread(image_path)
                if img is None:
                    continue
                    
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    face_img = gray[y:y + h, x:x + w]
                    # Resize face to uniform dimensions
                    face_img_resized = cv2.resize(face_img, FACE_SIZE)
                    
                    images.append(face_img_resized)
                    labels.append(label_counter)

        # Increment label counter ONLY after finishing an entire person's folder
        label_counter += 1

    if len(images) == 0:
        print("Error: No faces found in the training directory. Check your dataset!")
        return

    # Train the recognizer with the images and labels
    recognizer.train(images, np.array(labels))

    # Save the trained model
    recognizer.save("face_recognizer.yml")
    
    # Save label map to disk so recognize_faces can read it independently
    with open("labels_map.json", "w") as f:
        json.dump(label_map, f)

    global names
    names = label_map
    print("Training complete and model saved successfully.\n")


def recognize_faces():
    """
    Analyzes multiple people together via webcam, logs clean attendance logs,
    and automatically terminates if no recognized face is present for 3 seconds.
    """
    global names
    
    # Load the trained model
    if not os.path.exists("face_recognizer.yml"):
        print("Model file not found. Train the model first!")
        return
        
    recognizer.read("face_recognizer.yml")
    
    # Load label mapping from file
    if os.path.exists("labels_map.json"):
        with open("labels_map.json", "r") as f:
            # Convert JSON keys back to integers
            names = {int(k): v for k, v in json.load(f).items()}

    # Start the webcam
    cap = cv2.VideoCapture(0)

    # Get current date for file naming
    today = datetime.today().strftime('%Y-%m-%d')
    filename = f"{today}_Attendance.csv"

    # Helper function to cleanly overwrite CSV file and prevent duplicate history rows
    def save_attendance_to_csv(records):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "IN Time", "OUT Time"])  # Always write clean header
            for person, times in records.items():
                writer.writerow([person, times["IN"], times["OUT"]])

    # Structure to track live session attendance: { "Name": {"IN": "HH:MM:SS", "OUT": "HH:MM:SS"} }
    attendance = {}
    
    # Read existing file entries if you restart the program on the same day
    if os.path.exists(filename) and os.stat(filename).st_size > 0:
        try:
            with open(filename, mode='r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header row
                for row in reader:
                    if len(row) >= 3:
                        attendance[row[0]] = {"IN": row[1], "OUT": row[2]}
        except Exception:
            pass 

    # Initialize timer for the 3-second auto-termination feature
    last_seen_recognized_time = time.time()
    print("Camera activated. Scanning class...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        detected_names = set()  # Track who is in the *current* video frame
        any_recognized_in_frame = False

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            face_img_resized = cv2.resize(face_img, FACE_SIZE)
            
            label, confidence = recognizer.predict(face_img_resized)

            # Confidence threshold (lower values mean higher matching certainty in LBPH)
            if confidence < 90:
                name = names.get(label, "Unknown")

                if name != "Unknown":
                    any_recognized_in_frame = True
                    detected_names.add(name)
                    
                    # Scenario 1: New person enters for the first time
                    if name not in attendance:
                        current_time = datetime.now().strftime('%H:%M:%S')
                        attendance[name] = {"IN": current_time, "OUT": ""}
                        save_attendance_to_csv(attendance)
                    
                    # Scenario 2: Person returns to frame after checking out earlier
                    elif attendance[name]["OUT"] != "":
                        attendance[name]["OUT"] = ""  # Clear OUT time since they are back
                        save_attendance_to_csv(attendance)
                    
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({int(confidence)})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # --- AUTO TERMINATION TIMING LOGIC ---
        if any_recognized_in_frame:
            last_seen_recognized_time = time.time()  # Reset clock while a student is present
        else:
            if time.time() - last_seen_recognized_time >= 3.0:
                print("\n[Timeout] No recognized faces detected for 3 seconds. Auto-terminating system...")
                break

        # Scenario 3: Person leaves the frame (Check OUT tracking)
        file_changed = False
        for name in list(attendance.keys()):
            if name not in detected_names and attendance[name]["OUT"] == "":
                current_time = datetime.now().strftime('%H:%M:%S')
                attendance[name]["OUT"] = current_time
                file_changed = True
        
        # Rewrite file once per frame update loop if data changed
        if file_changed:
            save_attendance_to_csv(attendance)

        # Display window
        cv2.imshow('Classroom Attendance System', frame)

        # Break the loop immediately if user hits 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean resources up on shutdown
    cap.release()
    cv2.destroyAllWindows()


def main():
    base_path = r"Enter the path where these files are downloaded"

    # Train model (Can be commented out if you already trained your dataset once)
    train_model(base_path)

    # Start live scanner operation
    recognize_faces()


if __name__ == "__main__":
    main()
