
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from gtts import gTTS
import os
import time
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import ctypes
import pythoncom
import smtplib
from twilio.rest import Client
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import cv2
import numpy as np
import base64
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image, ImageDraw, ImageFilter
from botocore.exceptions import ClientError
import subprocess
from google.generativeai import GenerationModel

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_crop_face():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    time.sleep(5)  # Wait for 5 seconds before capturing the image

    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()
    if not ret:
        return None, "Failed to capture image"

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Crop the face from the frame
        face_crop = frame[y:y + h, x:x + w]
        
        # Encode the cropped image to base64 to send back to the frontend
        _, buffer = cv2.imencode('.png', face_crop)
        face_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return face_base64, None

    return None, "No face detected"

def apply_filter(image, filter_type):
    # Decode image
    image = Image.open(io.BytesIO(base64.b64decode(image)))
    img_array = np.array(image)
    
    if filter_type == 'blur':
        img_array = cv2.GaussianBlur(img_array, (15, 15), 0)
    elif filter_type == 'beauty':
        # Implement beauty filter (e.g., smoothing skin)
        img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
    elif filter_type == 'sunglasses':
        # Add sunglasses filter (placeholder for overlay logic)
        pass
    elif filter_type == 'cap':
        # Add cap filter (placeholder for overlay logic)
        pass
    elif filter_type == 'stars':
        # Add stars filter (placeholder for overlay logic)
        pass
    
    # Convert back to PIL Image
    image = Image.fromarray(img_array)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def scrape_google(query):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    query = query.replace(' ', '+')
    url = f"https://www.google.com/search?q={query}"
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print("Failed to retrieve page")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    
    results = []
    
    # Adjust the class name based on the latest inspection
    for g in soup.find_all('div', class_='g', limit=5):
        title_element = g.find('h3')
        link_element = g.find('a')

        if title_element and link_element:
            title = title_element.get_text()
            link = link_element['href']
            
            # Clean up Google’s link format
            if link.startswith('/url?q='):
                link = link.split('/url?q=')[1].split('&')[0]

            results.append({"title": title, "link": link})

    return results

# Twilio credentials
TWILIO_ACCOUNT_SID = 'ACab6b92bb7526e36ec96bd1e3012b7bd1'
TWILIO_AUTH_TOKEN = '69c2bca8b0d143f4be4f22ad4d21e8b0'
TWILIO_PHONE_NUMBER = '+19706446293'
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


def get_volume_interface():
    # Initialize COM
    pythoncom.CoInitialize()
    
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return ctypes.cast(interface, ctypes.POINTER(IAudioEndpointVolume))

# Your API key for Google Generative AI
api_key = "AIzaSyA9Z6-4-0tJOxV4JD1OxtTW9snfekxUavQ"

# Initialize the Google Generative AI client
gen_model = GenerationModel(api_key)

# Configuration for model generation
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# FLASK ROUTES

@app.route('/send_email', methods=['POST'])
def send_email():
    try:
        data = request.get_json()
        sender_email = data.get('sender_email')
        password = data.get('password')
        receiver_email = data.get('receiver_email')
        message = data.get('message')

        # Set up the SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)
        
        # Send the email
        email_body = f"Subject: New Message\n\n{message}"
        server.sendmail(sender_email, receiver_email, email_body)
        server.quit()

        return jsonify("Email sent successfully!"), 200
    except smtplib.SMTPAuthenticationError:
        return jsonify("Failed to authenticate. Check your email and password."), 401
    except Exception as e:
        return jsonify(f"An error occurred: {e}"), 500

@app.route('/send_sms', methods=['POST'])
def send_sms():
    data = request.get_json()
    to_number = data['to']
    message_body = data['message']

    try:
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=to_number
        )
        return jsonify({'status': 'success', 'sid': message.sid}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
@app.route('/connect_adb', methods=['POST'])
def send_adb():
    data = request.get_json()

    # Extract the recipient number and message from the JSON request
    to_number = data.get('to')
    message = data.get('message')

    # Validate the inputs
    if not to_number or not message:
        return jsonify({"status": "error", "message": "Recipient number and message are required."}), 400

    try:
        # Draft SMS via ADB
        draft_command = f'adb shell am start -a android.intent.action.SENDTO -d sms:{to_number} --es sms_body "{message}" --ez exit_on_sent true'
        os.system(draft_command)

        # Simulate pressing the "Send" button to send the SMS
        send_command = 'adb shell input keyevent 20'  # Key event for "Enter"
        os.system(send_command)
        send_command = 'adb shell input keyevent 20'  # Key event for "Enter"
        os.system(send_command)
        send_command = 'adb shell input keyevent 22'  # Key event for "Enter"
        os.system(send_command)
        send_command = 'adb shell input keyevent 22'  # Key event for "Enter"
        os.system(send_command)
        send_command = 'adb shell input keyevent 22'  # Key event for "Enter"
        os.system(send_command)
        send_command = 'adb shell input keyevent 66'  # Key event for "Enter"
        os.system(send_command)

        return jsonify({"status": "success", "message": "SMS sent successfully!"}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
  

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    results = scrape_google(query)
    return jsonify(results)

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    data = request.get_json()
    text = data['text']
    
    if text.strip() == "":
        return jsonify({"error": "No text provided"}), 400

    # Generate the TTS file
    try:
        tts = gTTS(text=text, lang='en')
        filename = f"speech_{int(time.time())}.mp3"  # Create a unique filename using a timestamp
        filepath = os.path.join('static', filename)
        tts.save(filepath)

        return jsonify({"speech_url": f"/static/{filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/<filename>', methods=['GET'])
def get_audio_file(filename):
    return send_from_directory('static', filename)
@app.route('/set_volume', methods=['POST'])
def set_volume():
    volume_level = request.json.get('volume')
    if volume_level is None:
        return jsonify({'error': 'No volume level provided'}), 400
    
    # Set the system volume
    volume_interface = get_volume_interface()
    volume_interface.SetMasterVolumeLevelScalar(int(volume_level) / 100, None)

    
    return jsonify({'success': True})

@app.route('/get_geo_location', methods=['GET'])
def get_geo_location():
    # Call the external API to get geolocation data
    response = requests.get("https://ipinfo.io")
    data = response.json()
    location = data['loc'].split(',')
    latitude = location[0]
    longitude = location[1]
    city = data.get('city')
    region = data.get('region')
    country = data.get('country')

    # Return the geolocation data as JSON
    return jsonify({
        'latitude': latitude,
        'longitude': longitude,
        'city': city,
        'region': region,
        'country': country
    })


@app.route('/get_volume', methods=['GET'])
def get_volume():
    volume_interface = get_volume_interface()
    current_volume = volume_interface.GetMasterVolumeLevelScalar() * 100
    return jsonify({'volume': current_volume})
@app.route('/send_bulk_email', methods=['POST'])
def send_bulk_email():
    try:
        data = request.get_json()
        sender_email = data.get('sender_email')
        password = data.get('password')
        receiver_emails = data.get('receiver_emails')  # List of emails
        message = data.get('message')

        # Set up the SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, password)

        # Email subject and message body
        email_body = f"Subject: New Message\n\n{message}"

        # Send email to each recipient
        for receiver_email in receiver_emails:
            server.sendmail(sender_email, receiver_email, email_body)

        server.quit()

        return jsonify("Bulk emails sent successfully!"), 200
    except smtplib.SMTPAuthenticationError:
        return jsonify("Failed to authenticate. Check your email and password."), 401
    except Exception as e:
        return jsonify(f"An error occurred: {e}"), 500

@app.route('/make_call', methods=['POST'])
def make_call():
    data = request.json
    to_phone = data.get('to_phone')

    try:
        call = client.calls.create(
            to=to_phone,
            from_=TWILIO_PHONE_NUMBER,
            url="http://demo.twilio.com/docs/voice.xml"
        )
        return jsonify({"status": "success", "call_sid": call.sid}), 200
    except Exception as e:
        return jsonify({"status": "failed", "message": str(e)}), 500

@app.route('/capture', methods=['POST'])
def capture():
    face_base64, error = detect_and_crop_face()
    
    if error:
        return jsonify({'status': 'error', 'message': error})
    
    return jsonify({'status': 'success', 'image': face_base64})

@app.route('/full_capture', methods=['POST'])
def full_capture():
    filter_type = request.form.get('filter')

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    time.sleep(5)  # Wait for 5 seconds to simulate timer

    # Capture the image from the webcam
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({'status': 'error', 'message': 'Failed to capture image'})

    # Apply the selected filter to the entire frame
    filtered_image = apply_filter(frame, filter_type)

    # Encode the filtered image to base64 format to send to the frontend
    _, buffer = cv2.imencode('.png', filtered_image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'status': 'success', 'image': image_base64})

@app.route('/apply-filter', methods=['POST'])
def apply_filter_route():
    try:
        data = request.json
        image = data.get('image')
        filter_type = data.get('filter')

        if not image or not filter_type:
            return jsonify({'error': 'Missing image or filter type'}), 400

        filtered_image = apply_filter(image, filter_type)
        if not filtered_image:
            return jsonify({'error': 'Invalid filter type'}), 400

        return jsonify({'image': filtered_image})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/launch-ec2', methods=['POST'])
def launch_ec2():
    data = request.json
    access_key = data.get('access_key')
    secret_key = data.get('secret_key')
    region = data.get('region')
    instance_type = data.get('instance_type')

    try:
        ec2 = boto3.client(
            'ec2',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )

        response = ec2.run_instances(
            ImageId='ami-12345678',  # Replace with a valid image ID for your region
            InstanceType=instance_type,
            MinCount=1,
            MaxCount=1
        )

        instance_id = response['Instances'][0]['InstanceId']
        return jsonify({'instance_id': instance_id})

    except ClientError as e:
        return jsonify({'error': str(e)}), 400

@app.route('/execute', methods=['POST'])
def execute_command():
    data = request.get_json()
    command = data.get('command')

    try:
        # Use subprocess.Popen to execute the command
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output, error = process.communicate()

        if process.returncode == 0:
            return jsonify({"output": output})
        else:
            return jsonify({"output": error, "error": True})
    except Exception as e:
        return jsonify({"output": str(e), "error": True})
@app.route('/upload', methods=['POST'])
def upload_file():
    # Load AWS credentials and S3 bucket name from environment variables
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

    # Ensure AWS credentials and bucket name are provided
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY or not S3_BUCKET_NAME:
        return jsonify({"error": "AWS credentials or S3 bucket name not provided"}), 500

    # Create an S3 client
    s3_client = boto3.client('s3',
                             aws_access_key_id=AWS_ACCESS_KEY_ID,
                             aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

    # Check if the file is present in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    # Ensure the file has a valid filename
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Upload the file to S3
        s3_client.upload_fileobj(file, S3_BUCKET_NAME, file.filename)
        return jsonify({"message": "File uploaded successfully"}), 200

    except Exception as e:
        return jsonify({"error": f"File upload failed: {str(e)}"}), 500
    
@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message', '')

        if not user_message:
            return jsonify({'error': 'Message is required'}), 400

        # Start chat session and get model response
        chat_session = gen_model.start_chat(
            model='gemini-1.5-flash', 
            generation_config=generation_config,
            history=[]
        )
        response = chat_session.send_message(user_message)

        return jsonify({'reply': response['text']})  # Send back the model's reply

    except Exception as e:
        print(f"Error processing chat request: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500



if __name__ == '__main__':
    app.run(port=80, host="0.0.0.0")
