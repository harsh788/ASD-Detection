from flask import *
import firebase_admin
from firebase_admin import credentials, storage
import numpy as np
import pandas as pd
import cv2
import os
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array

cred = credentials.Certificate("reasd532006-c2f0bf0ed2a9.json")
firebase_admin.initialize_app(cred,{'storageBucket': 'reasd532006.appspot.com'}) # connecting to firebase

app = Flask(__name__)

# Load the pre-trained face detector model and the ASD model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = tf.keras.models.load_model("base_model.keras")

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
        test_data, test_filenames = preprocess()
        images = predict(test_data, test_filenames)
        print(images)

        # Saving the images to the firebase storage
        for image in images:
            blob = bucket.blob(image)
            blob.upload_from_filename(f"output_faces/{image}")

        return render_template('index.html', message='File uploaded successfully')

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
        if frame_count == 200:
            break

    # Release the VideoCapture object
    cap.release()

    print("Total frames processed:", frame_count)
    print("Total faces detected:", faces_detected)

# Load and preprocess test data
def preprocess():
    test_data = []
    test_filenames = []
    test_dir = 'output_faces'

    for image_file in os.listdir(test_dir):
        image_path = os.path.join(test_dir, image_file)
        img = load_img(image_path, target_size=(224, 224))  # ResNet50 input size
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize pixel values
        test_data.append(img_array)
        test_filenames.append(image_file)

    test_data = np.array(test_data)
    print("Preprocessing complete")
    return test_data, test_filenames

def predict(test_data, test_filenames):
    predictions = model.predict(test_data)
    print("Predictions complete")

    images = []
    for i in range(len(predictions)):
        if predictions[i] >= 0.5:
            images.append(test_filenames[i])

    return images


if __name__ == '__main__':
    app.run(debug=True)