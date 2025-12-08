# Face Recognition System – Image Collection, Data Preparation, Training & Real-Time Inference

This repository provides an end-to-end pipeline for building a custom face-recognition system.
It includes tools for:
<ul>
  <li>Collecting face images from an IP camera or webcam</li>
  <li>Cleaning and consolidating captured data</li>
  <li>Training a deep-learning facial recognition model using MobileNetV2</li>
  <li>Running real-time face recognition in Google Colab</li>
</ul>
The system is modular, easy to extend, and suitable for small to medium-scale personal datasets.

<h2>📂 Project Structure</h2>

<table>
    <thead>
        <tr>
            <th>File Structure/Path</th>
            <th>Description/Comment</th>
        </tr>
    </thead>
    <tbody>
          <tr>
            <td><code>collect_data.py</code></td>
            <td><code>Capture face images from an IP webcam feed</code></td>
        </tr>
        <tr>
            <td><code>consolidated_data.py</code></td>
            <td><code>Preprocess, label &amp; serialize image dataset</code></td>
        </tr>
        <tr>
            <td><code>facedetect.py</code></td>
            <td><code>Model training (Designed for Google Colab)</code></td>
        </tr>
        <tr>
            <td><code>recognize.py</code></td>
            <td><code>Real-time recognition (Google Colab only)</code></td>
        </tr>
        <tr>
            <td><code>images</code></td>
            <td><code>Raw collected images</code></td>
        </tr>
        <tr>
            <td><code>clean_data</code></td>
            <td><code>Serialized dataset (images.p, labels.p)</code></td>
        </tr>
        <tr>
            <td><code>final_model.h5</code></td>
            <td><code>Trained model (generated after training)</code></td>
        </tr>
    </tbody>
</table>


<h2>🧠 Workflow Overview</h2>
<ol>
  <li>Data Collection – collect_data.py   
    
Runs locally to capture 100 face images per person from an IP or webcam feed.
</li>
  <li>Consolidate & Serialize – consolidated_data.py  
    
Runs locally to preprocess, resize, and save image data into .p files.
  </li>
  <li> Model Training – facedetect.py (Google Colab Recommended) 
    
This script is designed to be run inside Google Colab, taking advantage of:
<ul>
  <li>GPU acceleration</li>
  <li>Easy notebook-based visualization</li>
  <li>Larger memory limits</li>
</ul>

 The script performs:
  <ul>
    <li>Preprocessing</li>
    <li>Stratified data splitting</li>
    <li>MobileNetV2 training (head + fine-tuning)</li>
    <li>Checkpointing</li>
    <li>Evaluation</li>
  </ul>

  
  Outputs: final_model.h5 


  </li>


      
  <li>Real-Time Recognition – recognize.py (Google Colab ONLY)

 This script must be run in Google Colab because it:
 <ul>
   <li>Uses Google Colab's JavaScript bridge for webcam streaming</li>
   <li>Captures browser-based video frames</li>
   <li>Processes them through the trained model in real time</li>
   <li>Draws labels and bounding boxes on the live feed</li>
 </ul>
 
 Local Python environments cannot run this script due to the Colab-specific frontend integration.</li>
</ol>   


<h2>🚀 Getting Started</h2>  

   Prerequisites
   <ul>
     <li>Python 3.8+</li>
     <li>OpenCV</li>
     <li>TensorFlow/Keras</li>
     <li>NumPy</li>
     <li>Matplotlib</li>
     <li>Scikit-Learn</li>
   </ul>
  
   Install dependencies:
<pre><code>pip install tensorflow opencv-python scikit-learn matplotlib numpy</code></pre>


<h2>📸 Step-by-Step Usage</h2>

✔ Step 1: Collect Faces Locally
<pre><code>python collect_data.py</code></pre>

✔ Step 2: Generate Dataset Locally
<pre><code>python consolidated_data.py</code></pre>

✔ Step 3: Train Model in Google Colab

Upload:
<ul>
  <li><code>images.p</code></li>
  <li><code>labels.p</code></li>
  <li><code>facedetect.py</code></li>
</ul>



Then run:
<pre><code>!python facedetect.py</code></pre>


✔ Step 4: Real-Time Recognition in Google Colab

Upload:
<ul>
  <li><code>final_model.h5</code></li>
  <li>Haarcascade XML file</li>
</ul>


Run:
<pre><code>!python recognize.py</code></pre>


<h2>📊 Model Architecture</h2>
<ul>
  <li>MobileNetV2 backbone (pretrained on ImageNet)</li>
  <li>Global Average Pooling</li>
  <li>Dense(64, ReLU) + Dropout(0.5)</li>
  <li>Softmax classifier</li>
</ul>


Efficient, lightweight, and suitable for small custom datasets.


<h2>📝 Notes & Recommendations</h2>
<ul>
  <li>Ensure consistent lighting when collecting images</li>
  <li>Capture multiple angles for each person</li>
  <li>More data → better accuracy</li>
  <li>For security-grade use cases, consider larger datasets and more robust models</li>
</ul>


<h2>📄 License – The Unlicense</h2>

This project is released into the public domain under The Unlicense.You are free to copy, modify, distribute, or use the project for any purpose, without any conditions.

<hr>
