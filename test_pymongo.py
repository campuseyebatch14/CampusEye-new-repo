import requests

def send_email(name, student_id, branch, timestamp, photo_url):
    try:
        response = requests.post(
            'http://localhost:5000/send-email',
            json={
                'name': name,
                'studentId': student_id,
                'branch': branch,
                'timestamp': timestamp,
                'photoUrl': photo_url
            }
        )
        if response.status_code == 200:
            print(f"Email sent successfully via EmailJS for {name}")
        else:
            print(f"EmailJS failed: {response.text}")
    except Exception as e:
        print(f"Failed to send email via EmailJS: {e}")