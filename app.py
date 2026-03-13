from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
import cv2
import numpy as np
from dotenv import load_dotenv
import cloudinary
from cloudinary.uploader import upload
from cloudinary.exceptions import Error as CloudinaryError
import os
import requests
from io import StringIO, BytesIO
import csv
import pandas as pd

import mongo_utils
import model_utils

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
env_path = os.path.join(os.path.dirname(__file__), '.env')
if not os.path.exists(env_path):
    raise RuntimeError(".env file not found")

load_dotenv(env_path)
print(".env loaded successfully")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")

# Cloudinary Config
cloudinary.config(
    cloud_name=os.getenv("CLOUD_NAME"),
    api_key=os.getenv("API_KEY"),
    api_secret=os.getenv("API_SECRET")
)

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route('/')
def index():
    """Dashboard showing all registered students."""
    students_list = list(
        mongo_utils.students_collection.find({}, {'_id': 0, 'embedding': 0})
    )
    return render_template('index.html', students_list=students_list)


@app.route('/add-student', methods=['GET', 'POST'])
def add_student():
    """Manual registration for a single student."""
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
        # Upload to Cloudinary
        upload_result = upload(photo)
        photo_url = upload_result['secure_url']

        # Generate Face Embedding
        photo.seek(0)
        img = cv2.imdecode(np.frombuffer(photo.read(), np.uint8), cv2.IMREAD_COLOR)
        embedding = model_utils.getEmbedding(img)

        if embedding is None:
            flash('Clear face not detected. Upload a better photo.', 'error')
            return redirect(url_for('add_student'))

        if mongo_utils.students_collection.find_one({'studentId': student_id}):
            flash('Student ID already exists.', 'error')
            return redirect(url_for('add_student'))

        # Save to Database
        mongo_utils.students_collection.insert_one({
            'name': name,
            'studentId': student_id,
            'branch': branch,
            'embedding': embedding,
            'photoUrl': photo_url
        })

        flash('Student added successfully', 'success')
        return redirect(url_for('index'))

    except CloudinaryError as e:
        flash(f'Cloudinary error: {e}', 'error')
        return redirect(url_for('add_student'))


@app.route('/edit-student/<student_id>', methods=['GET', 'POST'])
def edit_student(student_id):
    """Update existing student details."""
    student = mongo_utils.getStudentDetails(student_id)

    if request.method == 'GET':
        return render_template('student_form.html', student=student)

    name = request.form['name']
    branch = request.form['branch']
    photo = request.files['photo']

    try:
        upload_result = upload(photo)
        photo_url = upload_result['secure_url']

        photo.seek(0)
        img = cv2.imdecode(np.frombuffer(photo.read(), np.uint8), cv2.IMREAD_COLOR)
        embedding = model_utils.getEmbedding(img)

        if embedding is None:
            flash('Clear face not detected.', 'error')
            return redirect(url_for('edit_student', student_id=student_id))

        mongo_utils.students_collection.update_one(
            {'studentId': student_id},
            {'$set': {
                'name': name,
                'branch': branch,
                'embedding': embedding,
                'photoUrl': photo_url
            }}
        )

        flash('Student updated successfully', 'success')
        return redirect(url_for('index'))

    except CloudinaryError as e:
        flash(f'Cloudinary error: {e}', 'error')
        return redirect(url_for('edit_student', student_id=student_id))


@app.route('/delete-student/<student_id>')
def delete_student(student_id):
    """Remove student from system."""
    mongo_utils.deleteStudent(student_id)
    flash('Student removed successfully', 'success')
    return redirect(url_for('index'))


# --------------------------------------------------
# ✅ ROBUST BULK UPLOAD ROUTE
# --------------------------------------------------
@app.route('/bulk-upload', methods=['GET', 'POST'])
def bulk_upload():
    if request.method == 'GET':
        return render_template('bulk_upload.html')

    file = request.files.get('file')
    if not file or file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('bulk_upload'))

    try:
        # Load the file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8-sig')
        else:
            df = pd.read_excel(file)

        # 1. Capture original names for the error message
        original_columns = list(df.columns)
        
        # 2. Clean column names: remove spaces, lowercase, and remove "full" to match both "Name" and "Full Name"
        df.columns = [str(c).strip().lower().replace(" ", "").replace("full", "") for c in df.columns]
        
        # 3. Updated required list (now just looking for 'name' instead of 'fullname')
        required = ['name', 'studentid', 'branch', 'imageurl']
        missing = [req for req in required if req not in df.columns]
        
        if missing:
            flash(f"Invalid columns. Found: {original_columns}. Missing: {missing}", 'error')
            return redirect(url_for('bulk_upload'))

        success_count = 0
        for _, row in df.iterrows():
            s_id = str(row['studentid']).strip()
            
            if mongo_utils.students_collection.find_one({'studentId': s_id}):
                continue

            try:
                img_response = requests.get(row['imageurl'], timeout=10)
                image_array = np.asarray(bytearray(img_response.content), dtype="uint8")
                img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                embedding = model_utils.getEmbedding(img)
                if embedding is None:
                    continue

                upload_result = upload(row['imageurl'])
                photo_url = upload_result['secure_url']

                mongo_utils.students_collection.insert_one({
                    'name': row['name'], # This will now work for both "name" or "Full Name"
                    'studentId': s_id,
                    'branch': row['branch'],
                    'embedding': embedding,
                    'photoUrl': photo_url
                })
                success_count += 1
            except Exception as e:
                print(f"Error processing {s_id}: {e}")

        flash(f'Successfully added {success_count} students.', 'success')
        return redirect(url_for('index'))

    except Exception as e:
        flash(f'Processing error: {str(e)}', 'error')
        return redirect(url_for('bulk_upload'))


# --------------------------------------------------
# ✅ AI CHATBOT ROUTE
# --------------------------------------------------
@app.route('/chat', methods=['POST'])
def chat():
    """Processes natural language queries about detection logs."""
    data = request.get_json()
    user_query = data.get('query', '').lower()
    
    try:
        detections = list(mongo_utils.detections_collection.find({}, {'_id': 0}))
        
        if not detections:
            return jsonify({'success': True, 'answer': "No detection records found yet."})
            
        df = pd.DataFrame(detections)
        df['timestamp'] = df['timestamp'].astype(str)
        
        all_names = df['name'].unique()
        found_name = None
        for name in all_names:
            if name.lower() in user_query:
                found_name = name
                break
        
        if found_name:
            student_logs = df[df['name'] == found_name]
            times = student_logs['timestamp'].tolist()
            count = len(times)
            if count > 5:
                recent = times[-5:]
                time_list = "<br>• " + "<br>• ".join(recent)
                response = f"Found {count} records for <b>{found_name}</b>.<br>Most recent:{time_list}"
            else:
                time_list = "<br>• " + "<br>• ".join(times)
                response = f"<b>{found_name}</b> was detected at:{time_list}"
            return jsonify({'success': True, 'answer': response})

        elif any(k in user_query for k in ['who', 'list', 'show', 'all']):
            names_str = ", ".join(df['name'].unique())
            return jsonify({'success': True, 'answer': f"Students detected: {names_str}."})

        elif any(k in user_query for k in ['count', 'how many']):
            count = df['studentId'].nunique()
            return jsonify({'success': True, 'answer': f"There are {count} unique students in the logs."})

        return jsonify({'success': True, 'answer': "I didn't understand. Try asking 'Who was seen?' or a name."})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# --------------------------------------------------
# ✅ DUAL NOTIFICATION EMAIL ROUTE
# --------------------------------------------------
@app.route('/send-email', methods=['POST'])
def send_email():
    """Sends automated alerts to both the recipient and admin email addresses."""
    live_image = request.files.get('live_image')
    data = request.form if live_image else (request.get_json() or {})

    if not data:
        return jsonify({'success': False, 'error': 'No data provided'})

    final_photo_url = data.get("photoUrl")

    if live_image:
        try:
            upload_result = upload(live_image)
            final_photo_url = upload_result['secure_url']
        except Exception as e:
            print(f"Cloudinary upload failed: {e}")

    # Dual Recipient Logic
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
                "photo_url": final_photo_url,  
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
        except requests.RequestException as e:
            print(f"Request error for {email_addr}: {e}")

    return jsonify({"success": overall_success})


@app.route('/download-report')
def download_report():
    """Generates a downloadable CSV of all detection records."""

