import os
import cv2
import numpy as np
import json
from datetime import datetime
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from tensorflow.keras.models import load_model
from deepface import DeepFace

class FaceDetector:
    def __init__(self):
        self.cap = None
        self.is_running = False
        
        # Load class labels - we know it's "a" and "b"
        self.labels = ["a", "b"]
        
        # Initialize balance for each person
        self.balances = {
            "a": 5000,
            "b": 5000
        }
        
        # Display names for UI (optional - you can customize these)
        self.display_names = {
            "a": "Person A",
            "b": "Person B"
        }
        
        # Load face cascade for detection (optional, can be removed)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def check_time_and_deduct(self, name):
        current_time = datetime.now().time()
        
        # Breakfast time (10:15 AM - 10:45 AM)
        breakfast_start = datetime.strptime("10:15", "%H:%M").time()
        breakfast_end = datetime.strptime("10:45", "%H:%M").time()
        
        # Lunch time (1:00 PM - 1:45 PM)
        lunch_start = datetime.strptime("13:00", "%H:%M").time()
        lunch_end = datetime.strptime("13:45", "%H:%M").time()

        if self.balances[name] <= 0:
            return "Insufficient funds, please refill"

        if breakfast_start <= current_time <= breakfast_end:
            if self.balances[name] >= 20:
                self.balances[name] -= 20
                display_name = self.display_names.get(name, name)
                return f"Deducted 20rs for breakfast for {display_name}. Balance: {self.balances[name]}"
            return "Insufficient funds, please refill"
            
        elif lunch_start <= current_time <= lunch_end:
            if self.balances[name] >= 50:
                self.balances[name] -= 50
                display_name = self.display_names.get(name, name)
                return f"Deducted 50rs for lunch for {display_name}. Balance: {self.balances[name]}"
            return "Insufficient funds, please refill"
            
        return "It is not lunch or snack time right now"

    def start_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
            else:
                self.is_running = False
    
    def stop_camera(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.is_running = False

    def verify_face(self, face_img):
        try:
            if face_img is None or face_img.size == 0:
                return "Unknown"

            # Save the face image temporarily
            temp_path = "temp_face.jpg"
            cv2.imwrite(temp_path, face_img)

            # Use DeepFace for face verification
            try:
                result = DeepFace.find(
                    img_path=temp_path,
                    db_path='database',
                    model_name='VGG-Face',
                    enforce_detection=False
                )

                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                if isinstance(result, list) and len(result) > 0 and not result[0].empty:
                    # Get the first match
                    first_match = result[0].iloc[0]
                    # Extract the identity path
                    identity_path = first_match['identity']
                    # Get the class name (a or b) from the path
                    person_name = identity_path.split(os.sep)[-2]
                    return person_name
                
                return "Unknown"
                
            except Exception as e:
                print(f"DeepFace error: {e}")
                return "Unknown"
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Unknown"

    def process_frame(self, frame):
        if not self.is_running:
            return frame, "Detector not ready"

        try:
            # Use DeepFace to extract faces directly from the frame
            faces = DeepFace.extract_faces(frame, enforce_detection=False)

            if faces:
                for face in faces:
                    # Get the face image and its coordinates
                    face_img = face['face']
                    facial_area = face['facial_area']
                    
                    # Extract coordinates from facial_area
                    if isinstance(facial_area, dict):
                        x = facial_area.get('x', 0)
                        y = facial_area.get('y', 0)
                        w = facial_area.get('w', 0)
                        h = facial_area.get('h', 0)
                    else:
                        # If facial_area is a numpy array or list
                        facial_area = np.array(facial_area).ravel()
                        if len(facial_area) >= 4:
                            x, y, w, h = facial_area[:4]
                        else:
                            continue
                    
                    person = self.verify_face(face_img)
                    
                    if person != "Unknown":
                        message = self.check_time_and_deduct(person)
                        display_name = self.display_names.get(person, person)
                        # Draw rectangle around the detected face
                        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                        cv2.putText(frame, f"{display_name}: {message}", (int(x), int(y-10)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Unknown", (int(x), int(y-10)),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                return frame, "Face(s) detected"
                
            return frame, "No recognized face detected"
            
        except Exception as e:
            print(f"Processing error: {e}")
            return frame, "Processing Error"

class FaceDetectorApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detector = FaceDetector()
        self.update_event = None

    def build(self):
        self.title = 'Face Detection System'
        
        root = BoxLayout(orientation='vertical')
        self.camera_widget = Image(allow_stretch=True, keep_ratio=False)
        root.add_widget(self.camera_widget)
        
        self.status_label = Label(
            text="Press START to begin detection",
            size_hint_y=None,
            height='48dp',
            font_size='20sp'
        )
        root.add_widget(self.status_label)
        
        self.control_button = Button(
            text="START DETECTION",
            size_hint_y=None,
            height='60dp',
            on_press=self.toggle_detection,
            background_color=(0.25, 0.7, 0.25, 1)
        )
        root.add_widget(self.control_button)
        
        return root

    def toggle_detection(self, instance):
        if self.detector.is_running:
            self.stop_detection(instance)
        else:
            self.start_detection(instance)

    def start_detection(self, instance):
        instance.text = "STOP DETECTION"
        instance.background_color = (0.7, 0.25, 0.25, 1)
        self.status_label.text = "Starting camera..."
        self.detector.start_camera()
        
        if self.detector.is_running:
            self.update_event = Clock.schedule_interval(self.update_camera, 1.0 / 30.0)
            self.status_label.text = "Detecting faces..."
        else:
            self.status_label.text = "Camera failed to start."
            self.control_button.text = "START DETECTION"
            self.control_button.background_color = (0.25, 0.7, 0.25, 1)

    def stop_detection(self, instance):
        instance.text = "START DETECTION"
        instance.background_color = (0.25, 0.7, 0.25, 1)
        self.status_label.text = "Detection stopped."
        self.detector.stop_camera()
        if self.update_event:
            Clock.unschedule(self.update_event)
            self.update_event = None

    def update_camera(self, dt):
        if not self.detector.is_running:
            return

        try:
            ret, frame = self.detector.cap.read()
            if ret:
                processed_frame, detection_text = self.detector.process_frame(frame)
                self.status_label.text = detection_text

                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_flipped = cv2.flip(frame_rgb, 0)
                buf = frame_flipped.tobytes()
                texture = Texture.create(
                    size=(frame_rgb.shape[1], frame_rgb.shape[0]),
                    colorfmt='rgb'
                )
                texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
                self.camera_widget.texture = texture
                
        except Exception:
            self.stop_detection(self.control_button)

    def on_stop(self):
        self.detector.stop_camera()
        if self.update_event:
            Clock.unschedule(self.update_event)

if __name__ == "__main__":
    FaceDetectorApp().run()