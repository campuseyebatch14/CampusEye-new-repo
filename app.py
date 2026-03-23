import os
import sys
import base64
from datetime import datetime

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
import pandas as pd

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

# Global State to prevent sending multiple emails for the same student in one session
notified_students = set()

# --------------------------------------------------
# 1. Dashboard & Management Routes
# --------------------------------------------------

@app.route('/')
def index():
    try:
        students_list = list(mongo_utils.students_collection.find({}, {'_id': 0, 'embedding': 0}))
        return render_template('index.html', students_list=students_list)
    except Exception as e:
        return f"Database Connection Error: {str(e)}", 500

@app.route('/add-student', methods=['GET', 'POST'])
def add_student():
    if request.method == 'GET':
        return render_template('student_form.html', student=None)
    
    try:
        name, s_id, branch = request.form['name'], request.form['student_id'], request.form['branch']
        photo = request.files['photo']
        
        upload_result = upload(photo)
        photo_url = upload_result['secure_url']

        photo.seek(0)
        img = cv2.imdecode(np.frombuffer(photo.read(), np.uint8), cv2.IMREAD_COLOR)
        embedding = model_utils.getEmbedding(img)

        if embedding is None:
            flash('Face not detected.', 'error')
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

@app.route('/edit-student/<student_id>', methods=['GET', 'POST'])
def edit_student(student_id):
    student = mongo_utils.getStudentDetails(student_id)
    if request.method == 'GET':
        return render_template('student_form.html', student=student)

    try:
        name, branch = request.form['name'], request.form['branch']
        photo = request.files.get('photo')
        update_data = {'name': name, 'branch': branch}

        if photo and photo.filename != '':
            upload_result = upload(photo)
            photo.seek(0)
            img = cv2.imdecode(np.frombuffer(photo.read(), np.uint8), cv2.IMREAD_COLOR)
            embedding = model_utils.getEmbedding(img)
            if embedding:
                update_data['embedding'] = embedding
                update_data['photoUrl'] = upload_result['secure_url']

        mongo_utils.students_collection.update_one({'studentId': student_id}, {'$set': update_data})
        flash('Student updated successfully', 'success')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Update error: {str(e)}', 'error')
        return redirect(url_for('edit_student', student_id=student_id))

@app.route('/delete-student/<student_id>')
def delete_student(student_id):
    try:
        mongo_utils.deleteStudent(student_id)
        flash('Student removed successfully', 'success')
    except Exception as e:
        flash(f'Delete error: {str(e)}', 'error')
    return redirect(url_for('index'))

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
                    'name': row['name'], 'studentId': s_id, 'branch': row['branch'],
                    'embedding': embedding, 'photoUrl': upload_result['secure_url']
                })
                success_count += 1
        flash(f'Successfully added {success_count} students.', 'success')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Processing error: {str(e)}', 'error')
        return redirect(url_for('bulk_upload'))

# --------------------------------------------------
# 2. AI Chatbot & Report Routes
# --------------------------------------------------

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('query', '').lower()
    try:
        logs = list(mongo_utils.detections_collection.find({}, {'_id': 0}))
        if not logs: return jsonify({'success': True, 'answer': "No logs found."})
        all_names = list(set(d['name'] for d in logs))
        found_name = next((n for n in all_names if n.lower() in query), None)
        if found_name:
            times = [d['timestamp'] for d in logs if d['name'] == found_name]
            return jsonify({'success': True, 'answer': f"<b>{found_name}</b> seen {len(times)} times. Recent: {times[-1]}"})
        return jsonify({'success': True, 'answer': "Ask about a specific student name."})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download-report')
def download_report():
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Name', 'ID', 'Branch', 'Timestamp'])
    logs = mongo_utils.detections_collection.find()
    for l in logs: cw.writerow([l.get('name'), l.get('studentId'), l.get('branch'), l.get('timestamp')])
    output = io.BytesIO(si.getvalue().encode())
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='report.csv')

# --------------------------------------------------
# 3. Email & Surveillance Logic
# --------------------------------------------------

@app.route('/send-email', methods=['POST'])
def send_email():
    """Sends automated alerts to both the recipient and admin email addresses."""
    data = request.get_json() or request.form
    
    recipients = [os.getenv("RECIPIENT_EMAIL"), os.getenv("ADMIN_EMAIL")]
    recipients = [email for email in recipients if email]
    
    overall_success = False

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
            response = requests.post(
                "https://api.emailjs.com/api/v1.0/email/send",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=10 
            )
            if response.status_code == 200:
                overall_success = True
        except Exception as e:
            print(f"Email error for {email_addr}: {e}")

    return jsonify({"success": overall_success})

@app.route('/surveillance')
def surveillance_page():
    """Renders the live camera control page."""
    return render_template('surveillance.html')

@app.route('/process-frame', methods=['POST'])
def process_frame():
    """Receives a frame, runs AI, logs to DB, and triggers email alerts."""
    try:
        data = request.get_json()
        image_data = data.get('image')
        header, encoded = image_data.split(",", 1)
        nparr = np.frombuffer(base64.decodebytes(encoded.encode()), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        res = model_utils.findSuspects(frame)
        found_ids = res['found_suspect_ids']
        
        if not found_ids:
            return jsonify({'success': True, 'matches': []})

        suspects_details = mongo_utils.getSuspectsDetails(found_ids)
        full_timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

        detection_records = []
        for suspect in suspects_details:
            s_id = suspect['studentId']
            
            # Trigger Email and DB Log only if not already notified in this session
            if s_id not in notified_students:
                notified_students.add(s_id)
                
                email_payload = {
                    'name': suspect['name'],
                    'studentId': s_id,
                    'branch': suspect['branch'],
                    'timestamp': full_timestamp,
                    'photoUrl': suspect['photoUrl']
                }
                
                # Trigger the email internally
                try:
                    requests.post('http://localhost:5000/send-email', json=email_payload, timeout=5)
                except:
                    pass

                # Prepare MongoDB log record
                detection_records.append({
                    'studentId': s_id,
                    'name': suspect['name'],
                    'branch': suspect['branch'],
                    'timestamp': full_timestamp,
                    'photoUrl': suspect['photoUrl']
                })

        # Store detection record in MongoDB
        if detection_records:
            mongo_utils.store_detection_records(detection_records)

        return jsonify({
            'success': True, 
            'matches': [s['name'] for s in suspects_details],
            'count': len(found_ids)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# --------------------------------------------------
# 4. Start Server
# --------------------------------------------------

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
