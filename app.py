import csv
from flask import Flask, render_template, jsonify
import threading
import os  # Added import

app = Flask(__name__)

# Import necessary libraries for YOLO detection
from ultralytics import YOLO
import cv2
import cvzone
import math

# Initialize YOLO model
model = YOLO("../Yolo-Weights/best.pt")

# Define class names
classNames = ["anger", "disgust", "fear", "happy", "neutral", "sad", "neutral"]

def detect_emotion_thread():
    cap = cv2.VideoCapture(0)  # For Webcam
    cap.set(3, 1280)
    cap.set(4, 720)

    while True:
        success, img = cap.read()
        if not success:
            break
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Class Name
                cls = int(box.cls[0])
                emotion_detected = classNames[cls]
                cap.release()
                cv2.destroyAllWindows()
                return emotion_detected  # Return the detected emotion

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    emotion_detected = detect_emotion_thread()  # Get the detected emotion
    return jsonify(emotion=emotion_detected)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/emotion/<emotion>')
def emotion_table(emotion):
    # Map emotion to corresponding CSV file
    csv_files = {
        "anger": "anger.csv",
        "disgust": "disgust.csv",
        "fear": "fear.csv",
        "happy": "happy.csv",
        "neutral": "neutral.csv",
        "sad": "sad.csv",
        "surprised": "surprised.csv"  # Assuming you have a surprised.csv file
    }
    if emotion in csv_files:
        csv_file = csv_files[emotion]
        with open(csv_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            data = list(reader)  # Convert the reader to a list of dictionaries
        return render_template('index.html', data=data)
    else:
        return "Emotion not supported"

if __name__ == '__main__':
    app.run(debug=True)
