from flask import Flask, request, render_template, redirect, url_for, Response, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import os
import cv2
import numpy as np
from keras.models import load_model
from collections import deque
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import ssl
import base64
import logging
from datetime import datetime
import requests

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

should_process_video = False


@app.route('/start_processing', methods=['POST'])
def start_processing():
    data = request.get_json()
    filename = data['filename']
    global should_process_video
    should_process_video = True
    return jsonify({'message': 'Processing started'})


@app.route('/')
def upload_file():
    return render_template('index.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)

        # Ensure the directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)

        # Return the filename as a plain text response
        return filename


@app.route('/preview/<filename>')
def preview(filename):
    return render_template('preview.html', filename=filename)


@app.route('/video_feed/<filename>')
def video_feed(filename):
    global should_process_video
    if not should_process_video:
        return '', 204
    return Response(generate_frames('static/uploads/' + filename),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/processed_video_feed/<filename>')
def processed_video_feed(filename):
    return Response(generate_frames('static/uploads/' + filename),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera_feed')
def camera_feed():
    return Response(generate_frames(0),  # 0 means capturing from the webcam
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def get_salesforce_access_token(client_id, client_secret, username, password, security_token=None, instance_url='https://login.salesforce.com'):
    """
    Retrieve a Bearer access token from Salesforce using OAuth 2.0 Username-Password Flow.
    
    Parameters:
    - client_id (str): Consumer Key from Salesforce Connected App.
    - client_secret (str): Consumer Secret from Salesforce Connected App.
    - username (str): Salesforce username (e.g., user@domain.com).
    - password (str): Salesforce password.
    - security_token (str, optional): Security token for the user (if required by org settings).
    - instance_url (str, optional): Salesforce instance URL (default is production login URL).
    
    Returns:
    - str: Access token (Bearer token) if successful.
    - None: If authentication fails, returns None and prints error.
    """
    try:
        # Construct the full password (append security token if provided)
        full_password = password
        if security_token:
            full_password += security_token
        
        # OAuth 2.0 endpoint for token request
        token_url = f"{instance_url}/services/oauth2/token"
        
        # Payload for Username-Password Flow
        payload = {
            'grant_type': 'password',
            'client_id': client_id,
            'client_secret': client_secret,
            'username': username,
            'password': full_password
        }
        
        # Send POST request to get access token
        response = requests.post(token_url, data=payload)
        
        # Check if request was successful
        if response.status_code == 200:
            token_data = response.json()
            access_token = token_data.get('access_token')
            print("Successfully retrieved Salesforce access token.")
            return access_token
        else:
            print(f"Failed to retrieve Salesforce access token: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"Error retrieving Salesforce access token: {str(e)}")
        return None

def send_notification_to_salesforce(frame):
    """
    Function to send a violence detection notification to Salesforce using a custom REST API endpoint.
    The frame is encoded as a base64 string and sent as part of the request.
    """
    try:
        # Get Salesforce credentials and configuration from environment variables
        client_id = os.getenv('SF_CLIENT_ID')
        client_secret = os.getenv('SF_CLIENT_SECRET')
        username = os.getenv('SF_USERNAME')
        password = os.getenv('SF_PASSWORD')
        security_token = os.getenv('SF_SECURITY_TOKEN')
        instance_url = os.getenv('SF_INSTANCE_URL', 'https://login.salesforce.com')
        
        # Get access token for Salesforce API
        access_token = get_salesforce_access_token(client_id, client_secret, username, password, security_token, instance_url)
        
        if not access_token:
            print("Cannot send notification: No access token available.")
            return False
        
        # Prepare data for the custom Salesforce API endpoint
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")   # Format as ISO with timezone
        #current_time = datetime.now()

        incident_data = {
            "Carema": "C0003",  # Replace with dynamic camera ID if needed
            "IssueDateTime": current_time,
            "Location": "",  # Replace with dynamic location if available
            "Origin": "Web",
            "Description": "Violence action was detected",
            "Account": "Dickenson plc",  # Replace with relevant Account name if needed
            "frame": base64.b64encode(cv2.imencode('.jpg', frame)[1].tobytes()).decode('utf-8')  # Encode frame as base64
        }
        
        # Salesforce custom REST API endpoint for ViolenceCase
        salesforce_endpoint = f"{instance_url}/services/apexrest/ViolenceCase/"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        # Send POST request to Salesforce
        response = requests.post(salesforce_endpoint, json=incident_data, headers=headers)
        
        # Check response status
        if response.status_code == 200:
            print("Notification sent to Salesforce successfully.")
            response_data = response.json()
            case_number = response_data.get("caseNumber", "N/A")
            print(f"Case Number: {case_number}")
            return True
        else:
            print(f"Failed to send notification to Salesforce. Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error sending notification to Salesforce: {str(e)}")
        return False

def generate_frames(video_path):
    try:
        # Debug: Check if the model file exists
        model_path = './model/3Model.h5'  # Updated to the new model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        print(f"Loading model from: {model_path}")
        model = load_model(model_path, compile=False)
        print("Model loaded successfully.")
        # Get the expected input shape from the model
        input_shape = model.input_shape
        print(f"Model expected input shape: {input_shape}")
        if len(input_shape) == 4:  # Single frame input like (None, height, width, channels)
            image_height, image_width = input_shape[1], input_shape[2]
            sequence_length = 1  # Single frame, not a sequence
        elif len(input_shape) == 5:  # Sequence input like (None, sequence_length, height, width, channels)
            sequence_length, image_height, image_width = input_shape[1], input_shape[2], input_shape[3]
        else:
            print("Unexpected input shape. Defaulting to 64x64 with sequence length 20.")
            sequence_length, image_height, image_width = 20, 64, 64
        print(f"Resizing frames to: {image_height}x{image_width}")
        print(f"Using sequence length: {sequence_length}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

    class_list = ["NonViolence", "Violence"]

    # Violence detection criteria
    VIOLENCE_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for violence detection
    VIOLENCE_FRAME_THRESHOLD = 20  # Number of consecutive frames with violence
    COOLDOWN_PERIOD_SECONDS = 300  # 5 minutes cooldown after a notification

    # Initialize video capture based on input
    if video_path == "camera":
        camera_url = "rtsp://[username]:[password]@[camera_ip_address]/stream"  # Replace with actual URL
        video_reader = cv2.VideoCapture(camera_url)
        if not video_reader.isOpened():
            raise ValueError(f"Could not open IP camera stream at {camera_url}")
        print(f"Connected to IP camera at {camera_url}")
    else:
        video_reader = cv2.VideoCapture(video_path)
        if not video_reader.isOpened():
            raise ValueError(f"Could not open video file at {video_path}")
    
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    print(f'The video has {fps} frames per second.')

    frames_queue = deque(maxlen=sequence_length)  # Use the dynamically set sequence_length
    violence_count = 0  # Counter for consecutive violence frames
    last_notification_time = None  # Track the last time a notification was sent
    predicted_class_name = ''
    predicted_confidence = 0

    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break
        # Resize frame to match model input
        resized_frame = cv2.resize(frame, (image_width, image_height))
        # Normalize the frame (if your model expects normalized input, e.g., 0-1 range)
        normalized_frame = resized_frame / 255.0
        # Add frame to queue
        frames_queue.append(normalized_frame)
        # Perform prediction only if queue is full
        if len(frames_queue) == sequence_length:
            frame_batch = np.expand_dims(np.array(frames_queue), axis=0)
            print(f"Frame batch shape (sequence) before prediction: {frame_batch.shape}")
            try:
                predicted_labels_probabilities = model.predict(frame_batch)[0]
                predicted_label = np.argmax(predicted_labels_probabilities)
                predicted_class_name = class_list[predicted_label]
                predicted_confidence = predicted_labels_probabilities[predicted_label]
            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                predicted_class_name = 'Error'
                predicted_confidence = 0
        # Draw prediction on frame
        if predicted_class_name:
            text = f"{predicted_class_name}: {predicted_confidence:.2%}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if predicted_class_name == "NonViolence" else (0, 0, 255), 2)
        # Handle alert logic (if applicable)
        current_time = datetime.now()

        if predicted_class_name == "Violence" and predicted_confidence > VIOLENCE_CONFIDENCE_THRESHOLD:
            violence_count += 1
            if violence_count >= VIOLENCE_FRAME_THRESHOLD:
                # Check if enough time has passed since the last notification
                if last_notification_time is None or (current_time - last_notification_time).total_seconds() > COOLDOWN_PERIOD_SECONDS:
                    print("Violence detected consistently! Sending notification...")
                    send_notification_to_salesforce(frame)  # Function to send data to Salesforce
                    last_notification_time = current_time
                violence_count = VIOLENCE_FRAME_THRESHOLD  # Cap the counter to avoid overflow
        else:
            violence_count = 0  # Reset counter if violence is not detected
        # Encode frame for streaming
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    video_reader.release()
    cv2.destroyAllWindows()





if __name__ == '__main__':
    app.run(debug=True)