# BLOCK 1: Install dependencies
# Run this block first to set up the required libraries
!pip install -q opencv-python-headless tensorflow numpy requests matplotlib

# BLOCK 2: Import libraries
# Run this block to import all necessary Python libraries
import cv2
import numpy as np
import time
import os
import requests
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from google.colab import files
from IPython.display import display, clear_output

# BLOCK 3: Download face detection model
# Run this block to download pretrained OpenCV face detection model
!mkdir -p models/face_detector
!wget -q -O models/face_detector/deploy.prototxt https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
!wget -q -O models/face_detector/res10_300x300_ssd_iter_140000.caffemodel https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

# BLOCK 4: Define helper models
# Run this block to define the model creation functions
def create_face_recognition_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(160, 160, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    model = Model(inputs=base_model.input, outputs=x)
    for layer in base_model.layers:
        layer.trainable = False
    return model

def create_threat_detection_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(4, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    for layer in base_model.layers:
        layer.trainable = False
    return model

# BLOCK 5: Define the ReconEye class
# Run this block to define the main ReconEye class
class ReconEye:
    def __init__(self, use_sample_data=True):
        print("Initializing ReconEye system...")
        self.confidence_threshold = 0.7
        
        # Load face detector
        self.face_detector = cv2.dnn.readNetFromCaffe(
            "models/face_detector/deploy.prototxt",
            "models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
        )
        
        # Load face recognition model
        print("Loading face recognition model...")
        self.face_recognizer = create_face_recognition_model()
        
        # Load threat detection model
        print("Loading threat detection model...")
        self.threat_detector = create_threat_detection_model()
        
        # Load or create sample known faces
        if use_sample_data:
            self.known_faces = self._create_sample_known_faces()
            
        print("ReconEye system initialized")
    
    def _create_sample_known_faces(self):
        """Create sample data for demonstration purposes"""
        return {
            "person1": {
                "embedding": np.random.rand(128),
                "metadata": {
                    "name": "John Doe",
                    "status": "wanted",
                    "threat_level": "high",
                    "case_number": "CR-2024-12345"
                }
            },
            "person2": {
                "embedding": np.random.rand(128),
                "metadata": {
                    "name": "Jane Smith",
                    "status": "person-of-interest",
                    "threat_level": "medium", 
                    "case_number": "CR-2024-67890"
                }
            }
        }
    
    def detect_faces(self, frame):
        """Detect faces in the frame"""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            1.0, 
            (300, 300), 
            (104.0, 177.0, 123.0)
        )
        
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                face = frame[startY:endY, startX:endX]
                
                if face.size > 0 and face.shape[0] > 0 and face.shape[1] > 0:
                    faces.append({
                        "face": face,
                        "coords": (startX, startY, endX, endY)
                    })
        
        return faces
    
    def recognize_face(self, face):
        """Generate embedding for a face and match against known faces database"""
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (160, 160))
        face_array = img_to_array(face_resized)
        face_preprocessed = preprocess_input(face_array)
        face_expanded = np.expand_dims(face_preprocessed, axis=0)
        
        embedding = self.face_recognizer.predict(face_expanded)[0]
        
        best_match = None
        best_distance = float('inf')
        
        for person_id, data in self.known_faces.items():
            known_embedding = data["embedding"]
            distance = np.linalg.norm(embedding - known_embedding)
            
            if distance < best_distance and distance < 0.6:
                best_distance = distance
                best_match = data["metadata"]
        
        return best_match, embedding
    
    def detect_threats(self, frame):
        """Detect potential threats in the frame"""
        frame_resized = cv2.resize(frame, (224, 224))
        frame_array = img_to_array(frame_resized)
        frame_preprocessed = preprocess_input(frame_array)
        frame_expanded = np.expand_dims(frame_preprocessed, axis=0)
        
        predictions = self.threat_detector.predict(frame_expanded)[0]
        
        threat_classes = ["normal", "suspicious", "weapon", "violent"]
        threat_index = np.argmax(predictions)
        threat_confidence = predictions[threat_index]
        
        if threat_index > 0 and threat_confidence > self.confidence_threshold:
            return {
                "type": threat_classes[threat_index],
                "confidence": float(threat_confidence),
                "timestamp": time.time()
            }
        
        return None
    
    def send_alert(self, alert_data):
        """Simulate sending alert to law enforcement server"""
        print("ALERT SENT:")
        print("-" * 40)
        for key, value in alert_data.items():
            print(f"{key}: {value}")
        print("-" * 40)
        return True
    
    def process_frame(self, frame):
        """Process a video frame for faces and threats"""
        display_frame = frame.copy()
        
        # Detect faces
        faces = self.detect_faces(frame)
        
        for face_data in faces:
            face = face_data["face"]
            (startX, startY, endX, endY) = face_data["coords"]
            
            # Recognize face
            match, _ = self.recognize_face(face)
            
            # Draw rectangle around face
            if match:
                if match["status"] == "wanted" or match["threat_level"] == "high":
                    color = (0, 0, 255)  # Red for high threat
                    
                    # Send alert for high-threat matches
                    alert_data = {
                        "alert_type": "wanted_person",
                        "person_data": match,
                        "frame_timestamp": time.time()
                    }
                    
                    self.send_alert(alert_data)
                else:
                    color = (0, 255, 255)  # Yellow for known but not high threat
            else:
                color = (0, 255, 0)  # Green for unknown face
            
            cv2.rectangle(display_frame, (startX, startY), (endX, endY), color, 2)
            
            # Add name label if matched
            if match:
                y = startY - 10 if startY - 10 > 10 else startY + 10
                text = f"{match['name']} - {match['status']}"
                cv2.putText(display_frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Detect threats
        threat = self.detect_threats(frame)
        
        if threat:
            # Display threat warning
            text = f"ALERT: {threat['type'].upper()} - {threat['confidence']:.2f}"
            cv2.putText(display_frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Send threat alert
            alert_data = {
                "alert_type": "threat_detected",
                "threat_data": threat
            }
            
            self.send_alert(alert_data)
        
        return display_frame

# BLOCK 6: Initialize the system
# Run this block to create an instance of ReconEye
reconeye = ReconEye()

# BLOCK 7: Upload and process an image
# Run this block to upload and process an image
def process_uploaded_image():
    print("Please upload an image...")
    uploaded = files.upload()
    
    if not uploaded:
        print("No image uploaded.")
        return
    
    # Get the filename of the uploaded image
    filename = next(iter(uploaded))
    
    # Read the image
    image = cv2.imread(filename)
    
    # Convert BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    processed_image = reconeye.process_frame(image)
    
    # Convert processed image to RGB for display
    processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    
    # Display original and processed images side by side
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image_rgb)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Processed Image")
    plt.imshow(processed_rgb)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Run this function to process an uploaded image
process_uploaded_image()

# BLOCK 8: Process a sample image from URL (alternative to uploading)
# Run this block if you want to use a sample image instead of uploading
def process_sample_image():
    # URL of a sample image
    url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/group.jpg"
    
    # Download the image
    !wget -q -O sample_image.jpg {url}
    
    # Read the image
    image = cv2.imread("sample_image.jpg")
    
    # Convert BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    processed_image = reconeye.process_frame(image)
    
    # Convert processed image to RGB for display
    processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    
    # Display original and processed images side by side
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image_rgb)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Processed Image")
    plt.imshow(processed_rgb)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Run this function to process a sample image
# process_sample_image()  # Uncomment this line to run