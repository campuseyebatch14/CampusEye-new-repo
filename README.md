
## Technologies Used

- **DeepFace Library:** Used for facial recognition.
- **Dlib:** Utilized for face detection.
- **Facenet:** Employed to obtain facial embeddings for recognition.
- **MongoDB:** Storage of student details.
- **Flask:** Web framework for creating the user interface.
- **HTML and CSS:** Used to design and style the web interface.
- **Multithreading:** Implemented for efficient real-time detection.

## Features

- Real-time facial recognition using a video stream.
- Telegram alerts for identified students with timestamp details.
- Database storage for student records.
- Web interface for managing student details.
- Multithreading for efficient real-time detection.

## Getting Started

### Prerequisites

- Ensure you have Python installed.
- Install required libraries using `requirements.txt`.

### Installation

1. Clone the repository: `git clone https://github.com/your-username/student-surveillance.git`
2. Navigate to the project directory: `cd student-surveillance`
3. Install dependencies: `pip install -r requirements.txt`

### Configuration
Set up your environment variables by following these steps:
1. Create a copy of the `.example.env` file and name it `.env`.
2. Open the newly created `.env` file and replace the placeholder values with your own.

### Usage
0. To start surveillance: `python main.py`
1. To start the web interface: `python app.py`
2. Access the web interface at `http://localhost:5000` to manage student records.


=======
CampusEye: Next-Gen Intelligent Surveillance System
===================================================

CampusEye is an advanced real-time surveillance and attendance management system that leverages face recognition technology to identify registered students, log their presence, and send automated alerts to administrators.

🚀 Key Features
---------------

*   **Real-time Face Recognition**: Utilizes the DeepFace library with Facenet to detect and identify faces from a live camera feed.
    
*   **Student Management Dashboard**: A web-based interface built with Flask to add, edit, and delete student profiles.
    
*   **Automated Alerts**: Sends instant email notifications with live captures of detected students via EmailJS.
    
*   **Attendance Logging**: Automatically records student presence in a local attendance.csv file and a MongoDB database.
    
*   **Report Generation**: Export detection records and attendance logs as CSV reports directly from the dashboard.
    
*   **Time-Slot Monitoring**: Surveillance logic can be configured to run only during specific designated hours.
    

🛠️ Technology Stack
--------------------

*   **Backend**: Python, Flask
    
*   **Database**: MongoDB (PyMongo)
    
*   **Computer Vision**: OpenCV, DeepFace (Facenet)
    
*   **Cloud Storage**: Cloudinary (for storing student photographs)
    
*   **Email Service**: EmailJS
    
*   **Frontend**: HTML5, CSS3
    

📋 Prerequisites
----------------

Before you begin, ensure you have the following accounts and credentials:

*   **Python 3.8+**
    
*   **MongoDB Atlas**: A connection URI for your database.
    
*   **Cloudinary**: Cloud name, API key, and API secret.
    
*   **EmailJS**: Service ID, Template ID, User ID, and Private Key.
    

⚙️ Installation
---------------

### 1\. Clone the Repository

Bash
git clone <repository-url>
cd CampusEye

### 2\. Create a Virtual Environment

Bash

python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate `

### 3\. Install Dependencies

Bash

pip install -r requirements.txt

The project requires several libraries including Flask, deepface, opencv-python, pymongo, and cloudinary.


🏃 Running the Application
--------------------------

### 1\. Start the Web Dashboard

Launch the Flask server to manage student registrations:

Bash

python app.py  

Access the dashboard at http://127.0.0.1:5000.

### 2\. Launch the Surveillance System

Run the main script to start real-time monitoring via your camera:

Bash

python main.py

_Note: Press 'q' to stop the camera feed._

📁 Project Structure
--------------------

*   app.py: Flask application for student management and report downloads.
    
*   main.py: The core surveillance script handling camera feed and detection logic.
    
*   model\_utils.py: Face embedding and representation utilities using DeepFace.
    
*   mongo\_utils.py: Database operations for MongoDB.
    
*   attendance.csv: Local log for recording student attendance timestamps.
    
*   templates/: HTML templates for the dashboard UI.
    
*   static/: CSS and image assets.
    

👥 Contributors
---------------

This project was developed by:

*   Reshmitha
    
*   Pavan
    
*   Yaswanth
    
*   Govind
>>>>>>> 2f609f86f133de1b530877bf4c1edd9e34d6631b
