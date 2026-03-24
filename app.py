import os
import sys
import base64
import io
import csv
import gc
from datetime import datetime

# --- CRITICAL: EMERGENCY RAM SAVERS (Keep at the very top) ---
# Force TensorFlow to use legacy mode and stop heavy logging to save memory
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

# Global State: Tracks notified students for the current session to prevent email spam
notified_students = set()

# --------------------------------------------------
# INTERNAL ALERT LOGIC (Prevents 502/Timeout Errors)
# --------------------------------------------------

def trigger_alert_internal(data):
    """Directly sends alerts to EmailJS without an internal HTTP request."""
    recipients = [os.getenv("RECIPIENT_EMAIL"), os.getenv("ADMIN_EMAIL")]
    recipients = [e for e in recipients if e]
    
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
                                json=payload, timeout=8)
            if res.status_code == 200: success = True
        except Exception as e:
            print(f"Alert failed for {email_addr}: {e}")
    return success

# --------------------------------------------------
# 1. DASHBOARD & STUDENT MANAGEMENT
# --------------------------------------------------

@app.route('/')
def index():
    try:
        # Fetch students but exclude embeddings to save memory
        students = list(mongo_utils.students_collection.find({}, {'_id': 0, 'embedding': 0}))
        return render_template('index.html', students_list=students, active_page='index')
    except Exception as e:
        return f"Database Error: {str(e)}", 500

@app.route('/add-student', methods=['GET', 'POST'])
def add_student():
    if request.method == 'GET':
        return render_template('student_form.html', student=None, active_page='student_form')
    
    try:
        name, s_id, branch = request.form['name'], request.form['student_id'], request.form['branch']
        f = request.files['photo']
        
        # Upload to Cloudinary
        up = upload(f)
        
        # Reset file pointer and read for AI processing
        f.seek(0)
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        emb = model_utils.getEmbedding(img)

        if emb is None:
            flash('Face not detected. Please try a clearer photo.', 'error')
            return redirect(url_for('add_student'))

        mongo_utils.students_collection.insert_one({
            'name': name, 'studentId': s_id, 'branch': branch, 
            'embedding': emb, 'photoUrl': up['secure_url']
        })
        
        flash('Student added successfully!', 'success')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('add_student'))

@app.route('/edit-student/<student_id>', methods=['GET', 'POST'])
def edit_student(student_id):
    student = mongo_utils.getStudentDetails(student_id)
    if request.method == 'GET':
        return render_template('student_form.html', student=student)
    try:
        mongo_utils.students_collection.update_one(
            {'studentId': student_id}, 
            {'$set': {'name': request.form['name'], 'branch': request.form['branch']}}
        )
        flash('Record updated!', 'success')
        return redirect(url_for('index'))
    except:
        return redirect(url_for('index'))

@app.route('/delete-student/<student_id>')
def delete_student(student_id):
    mongo_utils.deleteStudent(student_id)
    flash('Student record deleted.', 'success')
    return redirect(url_for('index'))

@app.route('/bulk-upload', methods=['GET', 'POST'])
def bulk_upload():
    if request.method == 'GET': return render_template('bulk_upload.html')
    file = request.files.get('file')
    if not file: return redirect(url_for('bulk_upload'))
    
    try:
        # Using CSV module instead of Pandas to save ~50MB RAM
        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        reader = csv.DictReader(stream)
        count = 0
        for row in reader:
            data = {k.lower().replace(" ", ""): v for k, v in row.items()}
            s_id = data.get('studentid')
            
            if not s_id or mongo_utils.students_collection.find_one({'studentId': s_id}):
                continue
                
            img_res = requests.get(data.get('imageurl'), timeout=10)
            img = cv2.imdecode(np.asarray(bytearray(img_res.content), dtype="uint8"), cv2.IMREAD_COLOR)
            emb = model_utils.getEmbedding(img)
            
            if emb is not None:
                up = upload(data.get('imageurl'))
                mongo_utils.students_collection.insert_one({
                    'name': data.get('name'), 'studentId': s_id, 'branch': data.get('branch'),
                    'embedding': emb, 'photoUrl': up['secure_url']
                })
                count += 1
        
        gc.collect() # Immediate cleanup
        flash(f'Bulk upload complete: {count} students added.', 'success')
        return redirect(url_for('index'))
    except Exception as e:
        gc.collect()
        flash(f'Bulk error: {str(e)}', 'error')
        return redirect(url_for('bulk_upload'))

# --------------------------------------------------
# 2. SURVEILLANCE & AI (BROWSER-BASED)
# --------------------------------------------------

@app.route('/surveillance')
def surveillance_page():
    return render_template('surveillance.html', active_page='surveillance')

@app.route('/process-frame', methods=['POST'])
def process_frame():
    """Receives browser frames, runs AI identification, logs to DB, and sends alerts."""
    try:
        data = request.get_json()
        raw = data.get('image').split(",", 1)[1]
        nparr = np.frombuffer(base64.b64decode(raw), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # AI Identification
        res = model_utils.findSuspects(frame)
        found_ids = res.get('found_suspect_ids', [])
        
        if not found_ids:
            del frame, nparr
            gc.collect()
            return jsonify({'success': True, 'detected': False, 'message': 'Scanning... No match'})

        suspects = mongo_utils.getSuspectsDetails(found_ids)
        ts = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        matches = []

        for s in suspects:
            s_id = s['studentId']
            matches.append(s['name'])
            
            if s_id not in notified_students:
                notified_students.add(s_id)
                alert_info = {
                    'name': s['name'], 'studentId': s_id, 
                    'branch': s['branch'], 'timestamp': ts, 
                    'photoUrl': s['photoUrl']
                }
                # Direct trigger
                trigger_alert_internal(alert_info)
                # Save log
                mongo_utils.store_detection_records([alert_info])

        # Mandatory memory cleanup
        del frame, nparr
        gc.collect()
        
        return jsonify({
            'success': True, 
            'detected': True, 
            'message': f"DETECTED: {', '.join(matches)}", 
            'matches': matches
        })
    except Exception as e:
        gc.collect()
        return jsonify({'success': False, 'error': str(e)})

# --------------------------------------------------
# 3. REPORTS & CHATBOT
# --------------------------------------------------

@app.route('/download-report')
def download_report():
    """Generates the attendance CSV required by the dashboard link."""
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Name', 'ID', 'Branch', 'Timestamp'])
    logs = mongo_utils.detections_collection.find()
    for l in logs:
        cw.writerow([l.get('name'), l.get('studentId'), l.get('branch'), l.get('timestamp')])
    output = io.BytesIO(si.getvalue().encode())
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='campus_report.csv')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('query', '').lower()
    try:
        logs = list(mongo_utils.detections_collection.find({}, {'_id': 0}))
        names = list(set(d['name'] for d in logs))
        found = next((n for n in names if n.lower() in query), None)
        
        if found:
            answer = f"<b>{found}</b> was recently detected by the system."
        else:
            answer = "I don't see that specific name in the recent detection logs."
            
        return jsonify({'success': True, 'answer': answer})
    except:
        return jsonify({'success': False, 'error': "Chatbot currently unavailable."})

# --------------------------------------------------
# 4. START SERVER
# --------------------------------------------------

if __name__ == '__main__':
    # Use 0.0.0.0 for Render compatibility
    app.run(host='0.0.0.0', port=5000)
