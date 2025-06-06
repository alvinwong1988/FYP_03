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
        model_path = './model/NewModel.h5'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        print(f"Loading model from: {model_path}")
        model = load_model(model_path, compile=False)
        print("Model loaded successfully.")
        # Get the expected input shape from the model
        input_shape = model.input_shape
        print(f"Model expected input shape: {input_shape}")
        # Extract height and width from input shape (assuming shape is (None, height, width, channels))
        if len(input_shape) == 4:  # Single frame input like (None, height, width, channels)
            image_height, image_width = input_shape[1], input_shape[2]
        elif len(input_shape) == 5:  # Sequence input like (None, sequence_length, height, width, channels)
            image_height, image_width = input_shape[2], input_shape[3]
        else:
            print("Unexpected input shape. Defaulting to 64x64.")
            image_height, image_width = 64, 64
        print(f"Resizing frames to: {image_height}x{image_width}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

    sequence_length = 20
    class_list = ["NonViolence", "Violence"]

    video_reader = cv2.VideoCapture(video_path)
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    print(f'The video has {fps} frames per second.')

    frames_queue = deque(maxlen=sequence_length)

    predicted_class_name = ''
    predicted_confidence = 0
    alart_count = 0
    mail_sent = False

    while video_reader.isOpened():
        ok, frame = video_reader.read()
        if not ok:
            break

        # Resize the frame to match the model's expected input size
        resized_frame = cv2.resize(frame, (image_width, image_height))  # cv2.resize takes (width, height)
        # Normalize the frame (values between 0 and 1)
        normalized_frame = resized_frame / 255.0
        # Append the normalized frame to the queue for sequence processing
        frames_queue.append(normalized_frame)

        if len(frames_queue) == sequence_length:
            # Check if model expects a single frame or a sequence
            if len(input_shape) == 4:
                # Single frame input
                frame_batch = np.expand_dims(normalized_frame, axis=0)
                print("Frame batch shape (single frame) before prediction:", frame_batch.shape)
            else:
                # Sequence input
                frame_batch = np.expand_dims(list(frames_queue), axis=0)
                print("Frame batch shape (sequence) before prediction:", frame_batch.shape)
            try:
                # Make prediction using the resized and normalized frame(s)
                predicted_labels_probabilities = model.predict(frame_batch)[0]
                predicted_label = np.argmax(predicted_labels_probabilities)
                predicted_class_name = class_list[predicted_label]
                predicted_confidence = predicted_labels_probabilities[predicted_label]
            except Exception as e:
                print(f"Error during prediction: {str(e)}")
                raise

        text = f'{predicted_class_name}: {predicted_confidence:.2f}'
        # Calculate the text size based on the video's height
        text_size = frame.shape[0] / 4  # Adjust the denominator to get the desired text size

        if predicted_class_name == "Violence":
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, text_size / 100, (0, 0, 255), 2)
            email_subject = 'Violence Detected!!!'
            email_body = '<p>We have detected violence in the video, please check.</p>'
            alart_count += 1
            if alart_count >= 10 and not mail_sent:
                # send_email(email_subject, email_body, frame)
                mail_sent = True
        else:
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, text_size / 100, (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    video_reader.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    app.run(debug=True)