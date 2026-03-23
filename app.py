import os
import sys
import base64
import io
import csv
import gc
from datetime import datetime

# --- EMERGENCY RAM SAVERS ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
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

def trigger_alert(data):
    """Direct EmailJS trigger."""
    recipients = [os.getenv("RECIPIENT_EMAIL"), os.getenv("ADMIN_EMAIL")]
    for email_addr in [e for e in recipients if e]:
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
        try: requests.post("https://api.emailjs.com/api/v1.0/email/send", json=payload, timeout=8)
        except: pass

@app.route('/')
def index():
    students = list(mongo_utils.students_collection.find({}, {'_id': 0, 'embedding': 0}))
    return render_template('index.html', students_list=students, active_page='index')

@app.route('/surveillance')
def surveillance_page():
    return render_template('surveillance.html', active_page='surveillance')

@app.route('/process-frame', methods=['POST'])
def process_frame():
    try:
        data = request.get_json()
        raw = data.get('image').split(",", 1)[1]
        nparr = np.frombuffer(base64.b64decode(raw), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        res = model_utils.findSuspects(frame)
        found_ids = res.get('found_suspect_ids', [])
        
        if not found_ids:
            del frame, nparr
            gc.collect()
            return jsonify({'success': True, 'detected': False, 'message': 'Scanning...'})

        suspects = mongo_utils.getSuspectsDetails(found_ids)
        ts = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        matches = []

        for s in suspects:
            s_id = s['studentId']
            matches.append(s['name'])
            if s_id not in notified_students:
                notified_students.add(s_id)
                info = {'name': s['name'], 'studentId': s_id, 'branch': s['branch'], 'timestamp': ts, 'photoUrl': s['photoUrl']}
                trigger_alert(info)
                mongo_utils.store_detection_records([info])

        del frame, nparr
        gc.collect()
        return jsonify({'success': True, 'detected': True, 'message': f"DETECTED: {', '.join(matches)}", 'matches': matches})
    except:
        gc.collect()
        return jsonify({'success': False})

@app.route('/add-student', methods=['GET', 'POST'])
def add_student():
    if request.method == 'GET': return render_template('student_form.html', student=None, active_page='student_form')
    try:
        f = request.files['photo']
        up = upload(f)
        f.seek(0)
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        emb = model_utils.getEmbedding(img)
        mongo_utils.students_collection.insert_one({
            'name': request.form['name'], 'studentId': request.form['student_id'], 
            'branch': request.form['branch'], 'embedding': emb, 'photoUrl': up['secure_url']
        })
        flash('Student added!', 'success')
        return redirect(url_for('index'))
    except: return redirect(url_for('add_student'))

@app.route('/download-report')
def download_report():
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Name', 'ID', 'Branch', 'Timestamp'])
    for l in mongo_utils.detections_collection.find():
        cw.writerow([l.get('name'), l.get('studentId'), l.get('branch'), l.get('timestamp')])
    return send_file(io.BytesIO(si.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name='report.csv')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
