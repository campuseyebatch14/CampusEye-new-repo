import os
import sys

# --- CRITICAL FIX FOR DEPLOYMENT ---
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

@app.route('/chat', methods=['POST'])
def chat():
    """RAM-efficient chatbot without using Pandas."""
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

@app.route('/send-email', methods=['POST'])
def send_email():
    """Handles alerts."""
    data = request.get_json() or request.form
    # Logic to call EmailJS or your email provider
    return jsonify({"success": True})

@app.route('/download-report')
def download_report():
    """Generates attendance CSV."""
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Name', 'ID', 'Branch', 'Timestamp'])
    logs = mongo_utils.detections_collection.find()
    for l in logs: cw.writerow([l.get('name'), l.get('studentId'), l.get('branch'), l.get('timestamp')])
    output = io.BytesIO(si.getvalue().encode())
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='report.csv')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
