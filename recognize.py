import cv2
import numpy as np
from keras.models import load_model

# Load Cascade Classifier
# Assuming haarcascade is in the same directory or provide absolute path if needed
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load Model
model = load_model('final_model.h5')

# Load Labels
try:
    labels = np.load("classes.npy")
    print("Loaded labels from classes.npy:", labels)
except:
    print("classes.npy not found, using default labels.")
    # Original labels from the user's project
    labels = ["abhisikta", "adhinayak", "aryak", "raj", "biswabarenya", "ram","shayam", "smruti ranjan"]

def get_pred_label(pred):
    if pred >= 0 and pred < len(labels):
        return labels[pred]
    return f"Unknown ({pred})"

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 100))
    img = cv2.equalizeHist(img)
    img = img.reshape(1, 100, 100, 1)
    img = img / 255.0
    return img

# Initialize Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
    
    faces = classifier.detectMultiScale(frame, 1.3, 5)
      
    for x, y, w, h in faces:
        face = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        try:
            pred = model.predict(preprocess(face), verbose=0)
            label = get_pred_label(np.argmax(pred))
            confidence = np.max(pred)
            
            label_text = f"{label} ({confidence:.2f})"
            
            cv2.putText(frame, label_text,
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 0, 0), 2)
        except Exception as e:
            print(f"Prediction Error: {e}")
        
    cv2.imshow("Face Recognition", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

