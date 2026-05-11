AI Utility Hub - Flask Backend
Overview

AI Utility Hub is a multi-functional Flask-based backend application that combines AI chat, multimedia processing, system utilities, communication services, cloud integrations, and computer vision features into a single platform.

The project provides REST APIs for:

AI-powered chat using Google Gemini
Text-to-speech conversion
Face detection and image filtering
SMS and call services using Twilio
Email and bulk email sending
AWS EC2 and S3 integrations
System volume control
Google web scraping
Command execution
Webcam image capture
Geolocation services
Features
AI Chat Assistant
Gemini AI integration
Interactive chatbot endpoint
Configurable generation parameters
Text-to-Speech
Convert text into speech using gTTS
Automatically generate MP3 files
Face Detection & Filters
Webcam face capture
Face cropping
Blur and beauty filters
Custom filter support
Communication Services
Send emails
Send bulk emails
Send SMS via Twilio
Make phone calls using Twilio
Send SMS through Android ADB
Cloud Services
Launch AWS EC2 instances
Upload files to AWS S3
System Utilities
Get and set system volume
Execute shell commands
Retrieve geolocation data
Search Utility
Google search scraping
Return top search results
Tech Stack
Backend
Python
Flask
Flask-CORS
AI & ML
Google Gemini API
OpenCV
NumPy
PIL
Cloud & APIs
AWS Boto3
Twilio API
Multimedia
gTTS
Matplotlib
Project Structure
project/
│
├── app.py
├── static/
│   └── generated mp3 files
├── requirements.txt
└── README.md
Installation
1. Clone Repository
git clone https://github.com/yourusername/ai-utility-hub.git

cd ai-utility-hub
2. Create Virtual Environment
Windows
python -m venv venv

venv\Scripts\activate
Linux / Mac
python3 -m venv venv

source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
Required Libraries
pip install flask flask-cors requests beautifulsoup4 gtts pycaw comtypes \
pythoncom twilio boto3 opencv-python numpy matplotlib pillow google-generativeai
Environment Variables

Create a .env file:

AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_BUCKET_NAME=your_bucket_name
Twilio Configuration

Replace these values inside the code:

TWILIO_ACCOUNT_SID = "your_sid"
TWILIO_AUTH_TOKEN = "your_token"
TWILIO_PHONE_NUMBER = "your_number"
Google Gemini Configuration

Replace your API key:

api_key = "YOUR_GEMINI_API_KEY"
Running the Application
python app.py

Server will start on:

http://0.0.0.0:80
API Endpoints
Email APIs
Send Email
POST /send_email
JSON Body
{
  "sender_email": "example@gmail.com",
  "password": "password",
  "receiver_email": "receiver@gmail.com",
  "message": "Hello"
}
Send Bulk Email
POST /send_bulk_email
SMS & Call APIs
Send SMS
POST /send_sms
Make Call
POST /make_call
AI Chat
Chat with Gemini
POST /chat
JSON Body
{
  "message": "Hello AI"
}
Webcam APIs
Capture Face
POST /capture
Full Capture with Filter
POST /full_capture
Audio APIs
Text to Speech
POST /text-to-speech
Get Volume
GET /get_volume
Set Volume
POST /set_volume
AWS APIs
Launch EC2
POST /launch-ec2
Upload File to S3
POST /upload
Utility APIs
Execute Terminal Command
POST /execute
Google Search
POST /search
Get Geo Location
GET /get_geo_location
Example Curl Requests
Chat API
curl -X POST http://localhost/chat \
-H "Content-Type: application/json" \
-d "{\"message\":\"Hello AI\"}"
Text to Speech
curl -X POST http://localhost/text-to-speech \
-H "Content-Type: application/json" \
-d "{\"text\":\"Hello World\"}"
Security Warning

This project currently contains highly sensitive operations such as:

Shell command execution
Email password handling
AWS credential handling
Twilio secrets
ADB command execution
Before Deployment

You should:

Use environment variables for secrets
Add authentication and authorization
Validate user inputs
Restrict command execution
Use HTTPS
Add rate limiting
Remove hardcoded credentials
Future Improvements
JWT Authentication
Docker Support
Kubernetes Deployment
Frontend Dashboard
Real-time Webcam Streaming
Advanced AI Assistant
Database Integration
Async Task Queue
License

This project is licensed under the MIT License.

Author

Developed using Flask, OpenCV, AWS, Twilio, and Google Gemini AI.
