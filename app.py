import os

# --- CRITICAL FIX FOR DEPLOYMENT ---
# Force TensorFlow to use Legacy Keras to prevent "ModuleNotFoundError: No module named 'tensorflow.keras'"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import cv2
import numpy as np
from dotenv import load_dotenv
import cloudinary
from cloudinary.uploader import upload
from cloudinary.exceptions import Error as CloudinaryError
import requests
import pandas as pd

# Import your custom modules
import mongo_utils
import model_utils

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")

# Cloudinary Config
cloudinary.config(
    cloud_name=os.getenv("CLOUD_NAME"),
    api_key=os.getenv("API_KEY"),
    api_secret=os.getenv("API_SECRET")
)

@app.route('/')
def index():
    students_list = list(mongo_utils.students_collection.find({}, {'_id': 0, 'embedding': 0}))
    return render_template('index.html', students_list=students_list)

@app.route('/add-student', methods=['GET', 'POST'])
def add_student():
    if request.method == 'GET':
        return render_template('student_form.html', student=None)

    name = request.form['name']
    student_id = request.form['student_id']
    branch = request.form['branch']
    photo = request.files['photo']

    if photo.filename == '':
        flash('Empty photo file', 'error')
        return redirect(url_for('add_student'))

    try:
        upload_result = upload(photo)
        photo_url = upload_result['secure_url']

        photo.seek(0)
        img = cv2.imdecode(np.frombuffer(photo.read(), np.uint8), cv2.IMREAD_COLOR)
        embedding = model_utils.getEmbedding(img)

        if embedding is None:
            flash('Clear face not detected. Upload a better photo.', 'error')
            return redirect(url_for('add_student'))

        if mongo_utils.students_collection.find_one({'studentId': student_id}):
            flash('Student ID already exists.', 'error')
            return redirect(url_for('add_student'))

        mongo_utils.students_collection.insert_one({
            'name': name,
            'studentId': student_id,
            'branch': branch,
            'embedding': embedding,
            'photoUrl': photo_url
        })

        flash('Student added successfully', 'success')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('add_student'))

@app.route('/bulk-upload', methods=['GET', 'POST'])
def bulk_upload():
    if request.method == 'GET':
        return render_template('bulk_upload.html')

    file = request.files.get('file')
    if not file:
        flash('No file selected', 'error')
        return redirect(url_for('bulk_upload'))

    try:
        df = pd.read_csv(file) if file.filename.endswith('.csv') else pd.read_excel(file)
        df.columns = [str(c).strip().lower().replace(" ", "").replace("full", "") for c in df.columns]
        
        success_count = 0
        for _, row in df.iterrows():
            s_id = str(row['studentid']).strip()
            if mongo_utils.students_collection.find_one({'studentId': s_id}):
                continue

            img_response = requests.get(row['imageurl'], timeout=10)
            img = cv2.imdecode(np.asarray(bytearray(img_response.content), dtype="uint8"), cv2.IMREAD_COLOR)
            embedding = model_utils.getEmbedding(img)

            if embedding is not None:
                upload_result = upload(row['imageurl'])
                mongo_utils.students_collection.insert_one({
                    'name': row['name'],
                    'studentId': s_id,
                    'branch': row['branch'],
                    'embedding': embedding,
                    'photoUrl': upload_result['secure_url']
                })
                success_count += 1

        flash(f'Successfully added {success_count} students.', 'success')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Processing error: {str(e)}', 'error')
        return redirect(url_for('bulk_upload'))

# ... [Keep your /chat and /send-email routes as they were] ...

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
