from flask import *
import firebase_admin
from firebase_admin import credentials, storage
import numpy as np
import pandas as pd
import cv2
import os

cred = credentials.Certificate("reasd532006-c2f0bf0ed2a9.json")
firebase_admin.initialize_app(cred,{'storageBucket': 'reasd532006.appspot.com'}) # connecting to firebase

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploader', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
    
        # file_path = "3.cpp"
        bucket = storage.bucket() # storage bucket
        blob = bucket.blob(f.filename) # creating a blob object
        blob.upload_from_string(
            f.stream.read(),
            content_type=f.content_type
        )

        detect_and_save_faces_in_video(f.filename, 'output_faces')

        return render_template('index.html', message='File uploaded successfully')

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_save_faces_in_video(video_path, output_dir):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through the frames
    frame_count = 0
    faces_detected = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Save frames with detected faces
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count}_face_{faces_detected}.jpg"), face_img)
            faces_detected += 1

        frame_count += 1

    # Release the VideoCapture object
    cap.release()

    print("Total frames processed:", frame_count)
    print("Total faces detected:", faces_detected)

if __name__ == '__main__':
    app.run(debug=True)