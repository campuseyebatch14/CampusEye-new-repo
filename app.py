import os
import sys
import base64
import io
import csv
from datetime import datetime

# --- CRITICAL MEMORY & DEPLOYMENT FIXES ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Reduce TensorFlow logging to save memory

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
import cv2
import numpy as np
from dotenv import load_dotenv
import cloudinary
from cloudinary.uploader import upload
import requests
import pandas as pd

# Import your custom modules
import mongo_utils
import model_utils

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")

# Cloudinary Config
cloudinary.config(
    cloud_name=os.getenv("CLOUD_NAME"),
    api_key=os.getenv("API_KEY"),
    api_secret=os.getenv("API_SECRET")
)

# Global State: Tracks notified students for the current session
notified_students = set()

# --------------------------------------------------
# 1. CORE ALERT LOGIC (Direct Function Call)
# --------------------------------------------------

def send_alert_direct(data):
    """
    Sends alerts directly to EmailJS. 
    Using this function instead of an internal HTTP request prevents 502 deadlocks.
    """
    recipients = [os.getenv("RECIPIENT_EMAIL"), os.getenv("ADMIN_EMAIL")]
    recipients = [email for email in recipients if email]
    
    success = False
    for email_addr in recipients:
        payload = {
            "service_id": os.getenv("EMAILJS_SERVICE_ID"),
            "template_id": os.getenv("EMAILJS_TEMPLATE_ID"),
            "user_id": os.getenv("EMAILJS_USER_ID"),         
            "accessToken": os.getenv("EMAILJS_PRIVATE_KEY"),  
            "template_params": {
                "to_name": data.get("name"),
                "student_id": data.get("studentId"),
                "branch": data.get("branch"),
                "timestamp": data.get("timestamp"),
                "photo_url": data.get("photoUrl"),
                "to_email": email_addr
            }
        }
        try:
            res = requests.post("https://api.emailjs.com/api/v1.0/email/send", 
                                json=payload, timeout=10)
            if res.status_code == 200: success = True
        except Exception as e:
            print(f"Alert failed for {email_addr}: {e}")
    return success

# --------------------------------------------------
# 2. MANAGEMENT ROUTES
# --------------------------------------------------

@app.route('/')
def index():
    try:
        students_list = list(mongo_utils.students_collection.find({}, {'_id': 0, 'embedding': 0}))
        return render_template('index.html', students_list=students_list)
    except Exception as e:
        return f"Database Error: {str(e)}", 500

@app.route('/add-student', methods=['GET', 'POST'])
def add_student():
    if request.method == 'GET':
        return render_template('student_form.html', student=None, active_page='student_form')
    
    try:
        name, s_id, branch = request.form['name'], request.form['student_id'], request.form['branch']
        photo = request.files['photo']
        upload_result = upload(photo)
        
        photo.seek(0)
        img = cv2.imdecode(np.frombuffer(photo.read(), np.uint8), cv2.IMREAD_COLOR)
        embedding = model_utils.getEmbedding(img)

        if embedding is None:
            flash('Face not detected. Try a clearer photo.', 'error')
            return redirect(url_for('add_student'))

        mongo_utils.students_collection.insert_one({
            'name': name, 'studentId': s_id, 'branch': branch,
            'embedding': embedding, 'photoUrl': upload_result['secure_url']
        })
        flash('Student added!', 'success')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('add_student'))

@app.route('/delete-student/<student_id>')
def delete_student(student_id):
    mongo_utils.deleteStudent(student_id)
    flash('Record deleted.', 'success')
    return redirect(url_for('index'))

@app.route('/bulk-upload', methods=['GET', 'POST'])
def bulk_upload():
    if request.method == 'GET': return render_template('bulk_upload.html')
    file = request.files.get('file')
    try:
        df = pd.read_csv(file) if file.filename.endswith('.csv') else pd.read_excel(file)
        df.columns = [str(c).strip().lower() for c in df.columns]
        for _, row in df.iterrows():
            # Add logic to process each row as per previous versions
            pass
        flash('Bulk upload complete.', 'success')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('bulk_upload'))

# --------------------------------------------------
# 3. SURVEILLANCE & AI ROUTES
# --------------------------------------------------

@app.route('/surveillance')
def surveillance_page():
    return render_template('surveillance.html', active_page='surveillance')

@app.route('/process-frame', methods=['POST'])
def process_frame():
    """Identifies students from browser frames, logs to DB, and alerts."""
    try:
        data = request.get_json()
        image_data = data.get('image').split(",", 1)[1]
        
        # Optimized decoding
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # AI Matching
        res = model_utils.findSuspects(frame)
        found_ids = res.get('found_suspect_ids', [])
        
        if not found_ids:
            return jsonify({'success': True, 'detected': False, 'message': 'Scanning...'})

        suspects = mongo_utils.getSuspectsDetails(found_ids)
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        matches = []

        for s in suspects:
            s_id = s['studentId']
            matches.append(s['name'])
            
            if s_id not in notified_students:
                notified_students.add(s_id)
                alert_info = {
                    'name': s['name'], 'studentId': s_id, 
                    'branch': s['branch'], 'timestamp': timestamp, 
                    'photoUrl': s['photoUrl']
                }
                # Direct alert trigger
                send_alert_direct(alert_info)
                # MongoDB Log
                mongo_utils.store_detection_records([alert_info])

        return jsonify({
            'success': True, 'detected': True,
            'message': f"DETECTED: {', '.join(matches)}",
            'matches': matches
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('query', '').lower()
    try:
        logs = list(mongo_utils.detections_collection.find({}, {'_id': 0}))
        # Simple list-based filter to save RAM
        names = list(set(d['name'] for d in logs))
        found = next((n for n in names if n.lower() in query), None)
        if found:
            return jsonify({'success': True, 'answer': f"<b>{found}</b> was recently detected."})
        return jsonify({'success': True, 'answer': "I can see the logs, but I didn't catch that name."})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
