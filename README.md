# Face Recognition System – Image Collection, Data Preparation, Training & Real-Time Inference

This repository provides a complete pipeline for building your own face-recognition system using custom data.
It includes:
<ul>
  <li>Local data collection using your laptop webcam</li>
  <li>Local dataset consolidation</li>
  <li>Model training (run in Google Colab)</li>
  <li>Local real-time face recognition using webcam</li>
</ul>
The system is lightweight, modular, and easy to extend for custom datasets.


<h2>📂 Project Structure</h2>

<table style="width:100%; border-collapse:collapse; font-family:Arial,Helvetica,sans-serif;">
  <thead>
    <tr style="background:#f4f4f4; text-align:left;">
      <th style="padding:10px; border:1px solid #ddd;">File / Folder</th>
      <th style="padding:10px; border:1px solid #ddd;">Purpose</th>
      <th style="padding:10px; border:1px solid #ddd;">Run Location</th>
      <th style="padding:10px; border:1px solid #ddd;">Notes / Path to check</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:10px; border:1px solid #ddd;"><code>collect_data.py</code></td>
      <td style="padding:10px; border:1px solid #ddd;">Collect face images using local webcam (saves cropped face images).</td>
      <td style="padding:10px; border:1px solid #ddd;">Local laptop</td>
      <td style="padding:10px; border:1px solid #ddd;">Writes to <code>images/</code>. Ensure webcam index in <code>cv2.VideoCapture(0)</code> is correct.</td>
    </tr>
    <tr>
      <td style="padding:10px; border:1px solid #ddd;"><code>consolidated_data.py</code></td>
      <td style="padding:10px; border:1px solid #ddd;">Convert images → resized grayscale dataset and pickle: <code>images.p</code>, <code>labels.p</code>.</td>
      <td style="padding:10px; border:1px solid #ddd;">Local laptop</td>
      <td style="padding:10px; border:1px solid #ddd;">Checks/edits: <code>img_dir</code> and <code>data_dir</code> variables (default: current working directory).</td>
    </tr>
    <tr>
      <td style="padding:10px; border:1px solid #ddd;"><code>facedetect.py</code></td>
      <td style="padding:10px; border:1px solid #ddd;">Model training script (expects pickled clean dataset). Trains a small CNN and saves model.</td>
      <td style="padding:10px; border:1px solid #ddd;"><strong>Google Colab</strong> (recommended/required)</td>
      <td style="padding:10px; border:1px solid #ddd;">Upload <code>clean_data/images.p</code> and <code>clean_data/labels.p</code> to Colab. Paths in script use <code>/content/</code>; update if needed. For larger datasets increase <code>epochs</code> and <code>batch_size</code>.</td>
    </tr>
    <tr>
      <td style="padding:10px; border:1px solid #ddd;"><code>recognize.py</code></td>
      <td style="padding:10px; border:1px solid #ddd;">Local real-time recognition using the trained model and webcam; overlays labels and confidence.</td>
      <td style="padding:10px; border:1px solid #ddd;">Local laptop</td>
      <td style="padding:10px; border:1px solid #ddd;">Ensure correct paths for Haarcascade, <code>final_model.h5</code>, and <code>classes.npy</code>. Update <code>cv2.VideoCapture</code> index if needed.</td>
    </tr>
    <tr>
      <td style="padding:10px; border:1px solid #ddd;"><code>images/</code> (folder)</td>
      <td style="padding:10px; border:1px solid #ddd;">Raw cropped face images collected locally by <code>collect_data.py</code>.</td>
      <td style="padding:10px; border:1px solid #ddd;">Local laptop</td>
      <td style="padding:10px; border:1px solid #ddd;">Filename format must be <code>&lt;name&gt;_&lt;index&gt;.jpg</code> for label extraction.</td>
    </tr>
    <tr>
      <td style="padding:10px; border:1px solid #ddd;"><code>clean_data/</code> (folder)</td>
      <td style="padding:10px; border:1px solid #ddd;">Holds the pickled datasets created by <code>consolidated_data.py</code>: <code>images.p</code>, <code>labels.p</code>.</td>
      <td style="padding:10px; border:1px solid #ddd;">Local laptop → upload to Colab for training</td>
      <td style="padding:10px; border:1px solid #ddd;">Confirm files exist before uploading to Colab; adjust paths in <code>facedetect.py</code> if needed.</td>
    </tr>
    <tr>
      <td style="padding:10px; border:1px solid #ddd;"><code>final_model.h5</code></td>
      <td style="padding:10px; border:1px solid #ddd;">Trained model output (produced by <code>facedetect.py</code>).</td>
      <td style="padding:10px; border:1px solid #ddd;">Produced in Colab → download to local laptop for <code>recognize.py</code></td>
      <td style="padding:10px; border:1px solid #ddd;">Place in same folder as <code>recognize.py</code> or update the path in that script.</td>
    </tr>
  </tbody>
</table>


<h2>⚠ Important Execution Guide (What Runs Where?)</h2>
<h4>✔ Local Laptop</h4>
These scripts run locally without needing Google Colab:
<ul>
  <li><code>collect_data.py</code></li>
  <li><code>consolidated_data.py</code></li>
  <li><code>recognize.py</code></li>
</ul>
You only need:
<pre><code>
  Python 3.8+
  OpenCV
  TensorFlow/Keras
  NumPy
</code></pre>

<h4>✔ Google Colab (Required for Training)</h4>

<code>facedetect.py</code> must be run in Google Colab.

Why?

<ul>
  <li>GPU acceleration is required for efficient training</li>
  <li>Script paths are written for Colab (/content/...)</li>
  <li>Training with CPU will be extremely slow</li>
</ul>

You must upload these files to Google Colab:
<pre><code>
    clean_data/images.p
    clean_data/labels.p
    facedetect.py
</code></pre>


<h2>🧠 Workflow Overview</h2>

<ol>
  <li><h4>Collect Training Data (Local) — <code>collect_data.py</code></h4>
    Captures 100 face images per person using your laptop camera.
    Output example:
    <pre>
      <code>images/john_0.jpg
      images/john_1.jpg
      ...
      </code>
    </pre>
  </li>
  <li><h4>Consolidate Dataset (Local) —<code>consolidated_data.py</code></h4>
    This script:
    <ul>
      <li>Loads images from /images</li>
      <li>Resizes to 100×100</li>
      <li>Converts to grayscale</li>
      <li>Creates pickled dataset files:</li>
      <ul>
        <li><code>clean_data/images.p</code></li>
        <li><code>clean_data/labels.p</code></li>
      </ul>
  </ul>

    
  ⚠ You may need to edit the folder paths:
  <pre>
    <code>img_dir = os.path.join(os.getcwd(), 'images')
    data_dir = os.path.join(os.getcwd(), 'clean_data')</code>
  </pre>
  If your paths differ, update them accordingly.
  </li>
  <li><h4>Train Model (Google Colab Only) — <code>facedetect.py</code></h4>
  Upload:
  <pre><code>images.p
labels.p
</code></pre>
  Run:
    <pre><code>!python facedetect.py </code> </pre>
  The model:
    <ul>
      <li>Uses a small LeNet-style CNN</li>
      <li>Designed for small datasets (a few hundred images total)</li>
    </ul>

    
  ⚠ Important Training Notes
  <ul>
    <li>For larger datasets, you must increase epochs and possibly batch size:
        
      
  Recommended adjustments:
      <pre><code>epochs = 30–50
batch_size = 32–64
</code></pre>
    </li>

   <li>Monitor validation accuracy to avoid overfitting.</li> 
   <li>If Colab crashes due to RAM, reduce batch size instead.</li>
  </ul>
  
  Output files:
  <pre>
    <code>
      final_model.h5
    </code>
  </pre>
Download these to your laptop for local recognition.
  </li>
  <li><h4>Run Real-Time Recognition (Local) — <code>recognize.py</code></h4>
   Requires:
   <ul>
     <li><code>final_model.h5</code></li>
     <li><code>classes.npy (optional but recommended)</code></li>
     <li>Haarcascade XML</li>
   </ul>
    
   ⚠ You may need to update paths here:
    <pre>
      <code>
         classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
         model = load_model('final_model.h5')
         labels = np.load("classes.npy")
      </code>
    </pre>
    If files are in another folder, update the paths accordingly.
    
  Run:
  <pre>
    <code>
      python recognize.py
    </code>
  </pre>
   
   Your webcam feed opens and displays predicted labels + confidence scores.
  </li>
</ol>


<h2>🛠 Required Path Adjustments (Very Important)</h2>
<ul>
  <li>In <code>consolidated_data.py</code>

  Ensure correct directories:
  <pre>
    <code>
      clean_data/
      images/
    </code>
  </pre>
  Modify if your project structure differs.</li>
  <li>In <code>recognize.py</code>

  Modify paths to:
  <ul>
    <li>Haarcascade file</li>
    <li>Model file</li>
    <li>Label file</li>
  </ul>
  Example:
  <pre>
    <code>
      classifier = cv2.CascadeClassifier('path/to/haarcascade.xml')
      model = load_model('path/to/final_model.h5')
      labels = np.load('path/to/classes.npy')
  </code>
  </pre></li>
  <li>In <code>facedetect.py</code>

Paths are written for Colab:
<pre><code>open('/content/images.p')
</code></pre>
If you rename the variables or folder, update them accordingly.
</li>
</ul>


<h2>🐞 Known Issues / Bugs</h2>

<ol>
  <li> Face detection sensitivity -> Low lighting or angled faces may fail detection.</li>
  <li>Data imbalance affects accuracy -> Ensure similar image count for each person.</li>
  <li>Hardcoded paths -> All three major scripts may need path corrections based on user environment.</li>
  <li>Webcam issues -> If <code> cv2.VideoCapture(0) </code>fails, try:
    <ul>
      <li><code>VideoCapture(1)</code></li>
      <li>Allow camera permissions</li>
      <li>Close other apps using camera</li></li>
    </ul>
</ol>


<h2>📄 License – The Unlicense</h2>

This project is released into the public domain under The Unlicense. You are free to copy, modify, distribute, or use the project for any purpose, without any conditions.

<hr>
