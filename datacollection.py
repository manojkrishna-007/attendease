import cv2
import os

# Initialize the face detector (Haar Cascade Classifier)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a folder to save the images (if it doesn't exist)
folder_name = input("Enter the person's name: ")
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Initialize the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Counter for the number of pictures taken
count = 0

print("Starting to capture images... Please ensure your face is clearly visible.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Convert the frame to grayscale (required for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop over all detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Save the image of the face
        face_img = frame[y:y + h, x:x + w]  # Crop the face region
        img_name = os.path.join(folder_name, f"img{count + 1}.jpg")
        cv2.imwrite(img_name, face_img)

        count += 1
        print(f"Image {count} saved!")

    # Display the resulting frame with face rectangle
    cv2.imshow('Capture Faces', frame)

    # Stop capturing after 500 images
    if count >= 1500:
        print("1500 images captured successfully.")
        break

    # Wait for a key press, if 'q' is pressed, exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
