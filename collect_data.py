import cv2
import numpy as np
import os

# Load Cascade Classifier
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

data = []
count = 0

# Create images directory if it doesn't exist
if not os.path.exists("images"):
    os.makedirs("images")

print("Enter label name after collecting data.")
print("Press 'q' to stop early, or wait for 100 samples.")

while len(data) < 100:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    
    faces = classifier.detectMultiScale(frame, 1.3, 5)
    
    if len(faces) > 0:
        for x, y, w, h in faces:
            # Crop face area
            face_frame = frame[y:y+h, x:x+w]
            cv2.imshow("Only face", face_frame)
            
            if len(data) < 100:
                data.append(face_frame)
                print(f"{len(data)}/100")
                
            # Draw rectangle on original frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            break # Only take one face per frame
            
    cv2.putText(frame, f"Count: {len(data)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("frame", frame)
    
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
        
if len(data) > 0:
    name = input("Enter Face holder name : ")
    for i in range(len(data)):
        cv2.imwrite(f"images/{name}_{i}.jpg", data[i])
    print(f"Saved {len(data)} images for {name}")
else:
    print("No data collected")

