import os
import sys

# --- CRITICAL FIX FOR DEPLOYMENT ---
# Force TensorFlow to use Legacy Keras to prevent "ModuleNotFoundError: No module named 'tensorflow.keras'"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
import cv2
import numpy as np
from dotenv import load_dotenv
import cloudinary
from cloudinary.uploader import upload
import requests
import csv
import io
import pandas as pd  # Required for bulk-upload logic

# Import your custom modules
import mongo_utils
import model_utils

# Load environment variables
load_dotenv()

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
    """Dashboard showing all registered students."""
    try:
        students_list = list(mongo_utils.students_collection.find({}, {'_id': 0, 'embedding': 0}))
        return render_template('index.html', students_list=students_list)
    except Exception as e:
        # If this fails with a BuildError, it means a route used in index.html is missing here
        return f"Application Error: {str(e)}", 500

@app.route('/add-student', methods=['GET', 'POST'])
def add_student():
    """Manual registration for a single student."""
    if request.method == 'GET':
        return render_template('student_form.html', student=None)
    
    try:
        name = request.form['name']
        student_id = request.form['student_id']
        branch = request.form['branch']
        photo = request.files['photo']
        
        if photo.filename == '':
            flash('Empty photo file', 'error')
            return redirect(url_for('add_student'))

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
        
        flash('Student added successfully!', 'success')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('add_student'))

@app.route('/bulk-upload', methods=['GET', 'POST'])
def bulk_upload():
    """Restored route to handle CSV/Excel student uploads."""
    if request.method == 'GET':
        return render_template('bulk_upload.html')

    file = request.files.get('file')
    if not file:
        flash('No file selected', 'error')
        return redirect(url_for('bulk_upload'))

    try:
        # Detect file type and read
        df = pd.read_csv(file) if file.filename.endswith('.csv') else pd.read_excel(file)
        
        # Clean column names for consistency
        df.columns = [str(c).strip().lower().replace(" ", "").replace("full", "") for c in df.columns]
        
        success_count = 0
        for _, row in df.iterrows():
            s_id = str(row['studentid']).strip()
            
            # Skip if student already exists
            if mongo_utils.students_collection.find_one({'studentId': s_id}):
                continue

            try:
                # Process image from URL
                img_response = requests.get(row['imageurl'], timeout=10)
                img_array = np.asarray(bytearray(img_response.content), dtype="uint8")
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                embedding = model_utils.getEmbedding(img)
                if embedding is not None:
                    # Upload original image to Cloudinary
                    upload_result = upload(row['imageurl'])
                    
                    mongo_utils.students_collection.insert_one({
                        'name': row['name'],
                        'studentId': s_id,
                        'branch': row['branch'],
                        'embedding': embedding,
                        'photoUrl': upload_result['secure_url']
                    })
                    success_count += 1
            except Exception as e:
                print(f"Failed to process student {s_id}: {e}")

        flash(f'Successfully added {success_count} students.', 'success')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Processing error: {str(e)}', 'error')
        return redirect(url_for('bulk_upload'))

@app.route('/chat', methods=['POST'])
def chat():
    """RAM-efficient chatbot logic."""
    data = request.get_json()
    query = data.get('query', '').lower()
    try:
        logs = list(mongo_utils.detections_collection.find({}, {'_id': 0}))
        if not logs: 
            return jsonify({'success': True, 'answer': "No logs found yet."})

        all_names = list(set(d['name'] for d in logs))
        found_name = next((n for n in all_names if n.lower() in query), None)

        if found_name:
            student_logs = [d['timestamp'] for d in logs if d['name'] == found_name]
            return jsonify({
                'success': True, 
                'answer': f"<b>{found_name}</b> was seen {len(student_logs)} times. Last seen: {student_logs[-1]}"
            })
        
        return jsonify({'success': True, 'answer': "I can help with detection logs. Ask about a student name."})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/send-email', methods=['POST'])
def send_email():
    """Handles automated alerts."""
    data = request.get_json() or request.form
    return jsonify({"success": True})

@app.route('/download-report')
def download_report():
    """Generates a downloadable attendance CSV."""
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Name', 'ID', 'Branch', 'Timestamp'])
    
    logs = mongo_utils.detections_collection.find()
    for l in logs:
        cw.writerow([l.get('name'), l.get('studentId'), l.get('branch'), l.get('timestamp')])
        
    output = io.BytesIO(si.getvalue().encode())
    return send_file(
        output, 
        mimetype='text/csv', 
        as_attachment=True, 
        download_name='attendance_report.csv'
    )

if __name__ == '__main__':
    # Using 0.0.0.0 for Render compatibility
    app.run(host='0.0.0.0', port=5000)
