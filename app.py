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
        # f.save(f.filename)
    
        # file_path = "3.cpp"
        bucket = storage.bucket() # storage bucket
        blob = bucket.blob(f.filename) # creating a blob object
        blob.upload_from_string(
            f.stream.read(),
            content_type=f.content_type
        )

        return render_template('index.html', message='File uploaded successfully')

if __name__ == '__main__':
    app.run(debug=True)