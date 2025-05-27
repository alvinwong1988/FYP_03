from flask import Flask, request, render_template, redirect, url_for, Response, jsonify
from werkzeug.utils import secure_filename
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
import datetime
import requests

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


def send_email(subject, body, attachment=None):
    try:
        # Define our SMTP email server details
        smtp_server = "smtp.gmail.com"
        port = 587  # For starttls
        username = ""
        password = ""

        # Create a secure SSL context
        context = ssl.create_default_context()

        # Try to log in to server and send email
        server = smtplib.SMTP(smtp_server, port)
        server.ehlo()  # Can be omitted
        server.starttls(context=context)  # Secure the connection
        server.ehlo()  # Can be omitted
        server.login(username, password)

        msg = MIMEMultipart()
        msg['From'] = username
        msg['To'] = 'winggoldgoldgold@S@gmail.com'
        #msg['To'] = 'mehedihasannirobcsediu@gmail.com'
        #msg['To'] = 'jubayermahmud12345@gmail.com'
        msg['Subject'] = subject
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Get the location
        response = requests.get('https://ipinfo.io')
        if response.status_code == 200:
            data = response.json()
            location = f"{data.get('city', 'Unknown city')}, {data.get('region', 'Unknown region')}"
        else:
            location = 'Unknown'

        # Create the email body
        body = (f'Dear Author,\n\nViolence Deteced at {current_time}.\nThe location is: {location}. \n\n'
                f'Please take an action.The current situation given bellow: ')

        msg.attach(MIMEText(body, 'plain'))

        if attachment is not None:
            _, img_encoded = cv2.imencode('.jpg', attachment, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            img_as_bytes = img_encoded.tobytes()

            img_part = MIMEBase('image', "jpeg")
            img_part.set_payload(img_as_bytes)
            encoders.encode_base64(img_part)

            img_part.add_header('Content-Disposition', 'attachment', filename='detected_frame.jpg')
            msg.attach(img_part)

        server.send_message(msg)
        server.quit()

        logging.info('Email sent.')
    except Exception as e:
        logging.error(f'Error sending email: {str(e)}')


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
    predicted_class_name = ''
    predicted_confidence = 0
    alart_count = 0
    mail_sent = False

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
        if predicted_class_name == "Violence" and predicted_confidence > 0.5:
            alart_count += 1
            if alart_count > 10 and not mail_sent:
                # Placeholder for email logic
                print("Violence detected! Sending alert...")
                mail_sent = True
        else:
            alart_count = 0
            mail_sent = False
        # Encode frame for streaming
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    video_reader.release()
    cv2.destroyAllWindows()





if __name__ == '__main__':
    app.run(debug=True)