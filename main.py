import cv2
import pytz
import threading
import sys
import time
import csv
from io import BytesIO
import requests
import os
from dotenv import load_dotenv
from datetime import datetime, time as time_type

import model_utils
import mongo_utils

# Debug: Log .env file path and contents
env_path = os.path.join(os.path.dirname(__file__), '.env')
print(f"Loading .env from: {env_path}")
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        print(f".env contents: {f.read()}")
else:
    print("Error: .env file not found")
load_dotenv(env_path)

TIME_ZONE = pytz.timezone('Asia/Kolkata')

TIME_SLOTS = [
   (time_type(7, 0), time_type(9, 0)),     # 8:00 AM - 9:00 AM
    (time_type(9, 00), time_type(10, 0)),   
    (time_type(10, 45), time_type(12, 00)),
    (time_type(12, 0), time_type(16, 0)),  
    (time_type(16, 0), time_type(20, 0)),   
    (time_type(23, 25), time_type(23, 59)),
    (time_type(20, 50), time_type(23, 59)),
    (time_type(1, 20), time_type(2, 0))     
]

# --------------------------------------------------
# Global State for "Email Once" & CSV Logic
# --------------------------------------------------
notified_students = set()
csv_lock = threading.Lock()  # Prevents file corruption when multiple threads write

def log_to_csv(name, student_id, branch, timestamp_str):
    """
    Updates attendance.csv.
    If student+date exists: appends timestamp to the row.
    If not: creates a new row.
    """
    filename = 'attendance.csv'
    current_date = datetime.now().strftime("%Y-%m-%d")
    rows = []
    found = False

    with csv_lock:  # Ensure only one thread writes at a time
        # 1. Read existing data
        if os.path.isfile(filename):
            with open(filename, 'r', newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    # Row format: Name, ID, Branch, Date, Time1, Time2...
                    # Check if this row belongs to the student AND the current date
                    if len(row) >= 4 and row[1] == student_id and row[3] == current_date:
                        if timestamp_str not in row:
                            row.append(timestamp_str)
                        found = True
                    rows.append(row)
        
        # 2. If not found, add new row
        if not found:
            # New row: Name, ID, Branch, Date, Time1
            rows.append([name, student_id, branch, current_date, timestamp_str])

        # 3. Write back to file
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
            print(f"Logged to CSV: {name} at {timestamp_str}")


def is_within_time_slots():
    """Check if current time is within any of the defined time slots."""
    current_time = datetime.now(TIME_ZONE).time()  
    for start_time, end_time in TIME_SLOTS:
        if start_time <= current_time <= end_time:
            return True
    return False

def check_frame(frame):
    try:
        res = model_utils.findSuspects(frame)
        found_suspect_ids = res['found_suspect_ids']
        suspects_img = res['suspects_img']

        num_suspects = len(found_suspect_ids)

        if num_suspects == 0:
            print('no match found')
            return

        print(num_suspects, 'matche(s) found')

        timestamp = time.strftime("%H:%M:%S") # Just time for the CSV columns
        full_timestamp = time.strftime("%d/%m/%Y %H:%M:%S")

        print("At:", full_timestamp)
        print('\n--------found ids---------')
        print(found_suspect_ids)
        print('--------------------------\n')

        # Encode the frame (with bounding box) to JPEG bytes
        _, img_encoded = cv2.imencode('.jpg', suspects_img)
        img_bytes = img_encoded.tobytes()

        suspects_details = mongo_utils.getSuspectsDetails(found_suspect_ids)
        
        detection_records = []
        for suspect in suspects_details:
            s_id = suspect['studentId']
            s_name = suspect['name']
            s_branch = suspect['branch']

            # 1. ALWAYS Log to CSV (Continuous)
            log_to_csv(s_name, s_id, s_branch, timestamp)

            # 2. CHECK if email already sent
            if s_id not in notified_students:
                # FIX: Add to set IMMEDIATELY to block other threads from sending duplicates
                # while this thread is busy uploading the image.
                notified_students.add(s_id)
                
                print(f"Sending First Alert for {s_name}...")
                
                caption = 'Student Name: {}\n Student Id: {}\n Branch: {}\n Found At: {}\n'.format(
                    s_name, s_id, s_branch, full_timestamp
                )
                print(caption)

                # Prepare data for Email
                data_payload = {
                    'name': s_name,
                    'studentId': s_id,
                    'branch': s_branch,
                    'timestamp': full_timestamp,
                    'photoUrl': suspect['photoUrl']
                }
                
                files_payload = {
                    'live_image': ('capture.jpg', img_bytes, 'image/jpeg')
                }

                recipient_email = os.getenv('RECIPIENT_EMAIL')
                if recipient_email:
                    data_payload['to_email'] = recipient_email
                    
                try:
                    response = requests.post(
                        'http://localhost:5000/send-email', 
                        data=data_payload, 
                        files=files_payload
                    )
                    print(f"Email Status: {response.status_code}")
                    
                    # If email failed, remove from set so we try again next time
                    if response.status_code != 200:
                        print("Email failed, removing from notified list to retry.")
                        notified_students.remove(s_id)
                        
                except requests.RequestException as e:
                    print(f"Failed to send email: {str(e)}")
                    notified_students.remove(s_id)

                # Log to MongoDB only on the FIRST detection (Email event)
                detection_records.append({
                    'studentId': s_id,
                    'name': s_name,
                    'branch': s_branch,
                    'timestamp': full_timestamp,
                    'photoUrl': suspect['photoUrl']
                })

        # Store detection record in MongoDB (Only for the email event)
        if detection_records:
            mongo_utils.store_detection_records(detection_records)

    except Exception as e:
        print(f"Error in check_frame: {e}")

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error opening camera")
    sys.exit()

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera", WINDOW_WIDTH, WINDOW_HEIGHT)

WAIT_DURATION = 4
FRAME_RATE = 30
target_frame = WAIT_DURATION * FRAME_RATE
frame_counter = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error in reading frame from camera")
        break

    # Only process frame if within specified time slots
    if is_within_time_slots():
        if frame_counter % target_frame == 0:
            try:
                threading.Thread(target=check_frame, args=(frame.copy(),), daemon=True).start()
            except Exception as e:
                print(f'Error in creating new thread: {e}')
    else:
        if frame_counter % 30 == 0: 
            print(f"Skipping detection: Time {datetime.now(TIME_ZONE).time()} outside slots")

    frame_counter += 1

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()