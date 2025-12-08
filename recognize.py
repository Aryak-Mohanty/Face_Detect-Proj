# =============================
# REAL WORKING WEBCAM FOR COLAB
# =============================
from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from base64 import b64decode
import cv2
import numpy as np
from keras.models import load_model
import time

# Load your cascade + model
classifier = cv2.CascadeClassifier("/content/haarcascade_frontalface_default (1).xml")
model = load_model("/content/final_model.h5")

# ---- Start webcam ----
js = Javascript("""
async function startWebcam() {
  const video = document.createElement('video');
  video.setAttribute('autoplay', '');
  video.setAttribute('playsinline', '');
  document.body.appendChild(video);

  const stream = await navigator.mediaDevices.getUserMedia({video: true});
  video.srcObject = stream;

  await new Promise(resolve => video.onloadedmetadata = resolve);
  window.webcamVideo = video;
}

async function captureFrame() {
  const video = window.webcamVideo;
  if (!video) return "";

  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);

  return canvas.toDataURL('image/jpeg', 0.9);
}

startWebcam();
""")
display(js)


# ======== FIXED PREPROCESS FUNCTION =========
# Model expects 160x160x3 RGB
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # convert BGR → RGB
    img = cv2.resize(img, (160, 160))              # FIXED SIZE
    img = img.astype("float32") / 255.0            # normalize
    img = img.reshape(1, 160, 160, 3)              # FIXED SHAPE
    return img


def get_pred_label(pred):
    labels = ["abhisikta", "adhinayak", "aryak", "raj",
              "biswabarenya", "ram", "shayam", "smruti ranjan"]
    return labels[pred]


# ======== MAIN LOOP (unchanged) =========
print("Webcam streaming...")

while True:
    data = ""
    while data == "":
        data = eval_js("captureFrame()")
        time.sleep(0.02)

    img_bytes = b64decode(data.split(',')[1])
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    faces = classifier.detectMultiScale(frame, 1.5, 5)

    for x, y, w, h in faces:
        face = frame[y:y+h, x:x+w]
        pred = np.argmax(model.predict(preprocess(face)))

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 5)
        cv2.putText(frame, get_pred_label(pred), (x, y-10),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    _, jpeg = cv2.imencode('.jpg', frame)
    display(Image(data=jpeg.tobytes()))

    time.sleep(0.05)
