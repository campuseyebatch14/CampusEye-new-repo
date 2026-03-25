import os
import sys
import base64
import io
import csv
import gc
from datetime import datetime

# --- EMERGENCY RAM SAVERS (MUST BE AT TOP) ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1" 

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
import cv2
import numpy as np
from dotenv import load_dotenv
import cloudinary
from cloudinary.uploader import upload
import requests

import mongo_utils
import model_utils

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")

cloudinary.config(
    cloud_name=os.getenv("CLOUD_NAME"),
    api_key=os.getenv("API_KEY"),
    api_secret=os.getenv("API_SECRET")
)

notified_students = set()

def trigger_alert_now(data):
    recipients = [os.getenv("RECIPIENT_EMAIL"), os.getenv("ADMIN_EMAIL")]
    for email_addr in [e for e in recipients if e]:
        payload = {
            "service_id": os.getenv("EMAILJS_SERVICE_ID"),
            "template_id": os.getenv("EMAILJS_TEMPLATE_ID"),
            "user_id": os.getenv("EMAILJS_USER_ID"),         
            "accessToken": os.getenv("EMAILJS_PRIVATE_KEY"),  
            "template_params": {
                "to_name": data.get("name"), "student_id": data.get("studentId"),
                "branch": data.get("branch"), "timestamp": data.get("timestamp"),
                "photo_url": data.get("photoUrl"), "to_email": email_addr
            }
        }
        try: requests.post("https://api.emailjs.com/api/v1.0/email/send", json=payload, timeout=8)
        except: pass

# --------------------------------------------------
# 1. UPDATED ADD STUDENT (Memory Safe)
# --------------------------------------------------
@app.route('/add-student', methods=['GET', 'POST'])
def add_student():
    if request.method == 'GET':
        return render_template('student_form.html', student=None, active_page='student_form')
    
    # Use .get() to prevent "400 Bad Request" if a field is missing
    name = request.form.get('name')
    s_id = request.form.get('student_id')
    branch = request.form.get('branch')
    photo = request.files.get('photo')

    if not all([name, s_id, branch, photo]) or photo.filename == '':
        flash('All fields and a photo are required.', 'error')
        return redirect(url_for('add_student'))

    try:
        # 1. Cloudinary Upload
        up = upload(photo)
        
        # 2. Extract AI Features (The OOM Danger Zone)
        photo.seek(0)
        img_bytes = np.frombuffer(photo.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        
        # RESIZE TO 160x160: Drastically reduces RAM usage of TensorFlow
        img_small = cv2.resize(img, (160, 160))
        
        emb = model_utils.getEmbedding(img_small)

        if emb is None:
            del img, img_bytes, img_small
            gc.collect()
            flash('Face not detected. Try a clearer photo.', 'error')
            return redirect(url_for('add_student'))

        # 3. DB Save
        mongo_utils.students_collection.insert_one({
            'name': name, 'studentId': s_id, 'branch': branch, 
            'embedding': emb, 'photoUrl': up['secure_url']
        })
        
        # 4. Aggressive Cleanup
        del img, img_bytes, img_small, emb
        gc.collect()
        
        flash(f'Student {name} added successfully!', 'success')
        return redirect(url_for('index'))
    except Exception as e:
        gc.collect()
        flash(f'Server Busy (OOM risk). Try again in a minute.', 'error')
        return redirect(url_for('add_student'))

@app.route('/')
def index():
    try:
        students = list(mongo_utils.students_collection.find({}, {'_id': 0, 'embedding': 0}))
        return render_template('index.html', students_list=students, active_page='index')
    except: return "DB Error", 500

@app.route('/surveillance')
def surveillance_page():
    return render_template('surveillance.html', active_page='surveillance')

@app.route('/process-frame', methods=['POST'])
def process_frame():
    try:
        data = request.get_json()
        if not data or 'image' not in data: return jsonify({'success': False})
        
        raw = data.get('image').split(",", 1)[1]
        nparr = np.frombuffer(base64.b64decode(raw), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Resize frame for surveillance scanning to save RAM
        frame_small = cv2.resize(frame, (320, 240))

        res = model_utils.findSuspects(frame_small)
        found_ids = res.get('found_suspect_ids', [])
        
        if not found_ids:
            del frame, frame_small, nparr
            gc.collect()
            return jsonify({'success': True, 'detected': False, 'message': 'Scanning...'})

        suspects = mongo_utils.getSuspectsDetails(found_ids)
        ts = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        matches = [s['name'] for s in suspects]

        for s in suspects:
            if s['studentId'] not in notified_students:
                notified_students.add(s['studentId'])
                trigger_alert_now({'name': s['name'], 'studentId': s['studentId'], 'branch': s['branch'], 'timestamp': ts, 'photoUrl': s['photoUrl']})
                mongo_utils.store_detection_records([{'studentId': s['studentId'], 'name': s['name'], 'branch': s['branch'], 'timestamp': ts, 'photoUrl': s['photoUrl']}])

        del frame, frame_small, nparr
        gc.collect()
        return jsonify({'success': True, 'detected': True, 'message': f"DETECTED: {', '.join(matches)}", 'matches': matches})
    except:
        gc.collect()
        return jsonify({'success': False})

@app.route('/download-report')
def download_report():
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Name', 'ID', 'Branch', 'Timestamp'])
    for l in mongo_utils.detections_collection.find():
        cw.writerow([l.get('name'), l.get('studentId'), l.get('branch'), l.get('timestamp')])
    return send_file(io.BytesIO(si.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name='report.csv')

@app.route('/edit-student/<student_id>', methods=['GET', 'POST'])
def edit_student(student_id):
    student = mongo_utils.getStudentDetails(student_id)
    if request.method == 'GET': return render_template('student_form.html', student=student)
    mongo_utils.students_collection.update_one({'studentId': student_id}, {'$set': {'name': request.form.get('name'), 'branch': request.form.get('branch')}})
    return redirect(url_for('index'))

@app.route('/delete-student/<student_id>')
def delete_student(student_id):
    mongo_utils.deleteStudent(student_id)
    return redirect(url_for('index'))

@app.route('/chat', methods=['POST'])
def chat():
    return jsonify({'success': True, 'answer': "Chatbot ready."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1000)
