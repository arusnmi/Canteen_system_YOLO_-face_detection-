import os
import cv2
import numpy as np
from datetime import datetime
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from deepface import DeepFace

class FaceDetector:
    def __init__(self):
        self.cap = None
        self.is_running = False
        
        # Database paths for the two people
        self.database_a = "database/a"
        self.database_b = "database/b"
        
        # Initialize balance for each person
        self.balances = {
            "Person A": 5000,
            "Person B": 5000
        }
        
        # Load face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def check_time_and_deduct(self, name):
        current_time = datetime.now().time()
        
        # Check for breakfast time (10:15 AM - 10:45 AM)
        breakfast_start = datetime.strptime("10:00", "%H:%M").time()
        breakfast_end = datetime.strptime("11:00", "%H:%M").time()
        
        # Check for lunch time (1:00 PM - 1:45 PM)
        lunch_start = datetime.strptime("12:45", "%H:%M").time()
        lunch_end = datetime.strptime("14:00", "%H:%M").time()

        snacks_start = datetime.strptime("15:00", "%H:%M").time()
        snacks_end = datetime.strptime("16:00", "%H:%M").time()

        if self.balances[name] <= 0:
            return "Insufficient funds, please refill"

        if breakfast_start <= current_time <= breakfast_end:
            if self.balances[name] >= 74:
                self.balances[name] -= 74
                return f"Deducted 74 rs for breakfast for {name}. Balance: {self.balances[name]}"
            return "Insufficient funds, please refill"
            
        elif lunch_start <= current_time <= lunch_end:
            if self.balances[name] >= 115:
                self.balances[name] -= 115
                return f"Deducted 115 rs for lunch for {name}. Balance: {self.balances[name]}"
            return "Insufficient funds, please refill"
        elif snacks_start <= current_time <= snacks_end:
            if self.balances[name] >= 64:
                self.balances[name] -= 64
                return f"Deducted 64 rs for snacks for {name}. Balance: {self.balances[name]}"
            return "Insufficient funds, please refill"
        return "It is not breakfast, lunch or snack time right now"

    def start_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                print("Camera started.")
            else:
                print("Failed to open camera.")
                self.is_running = False
    
    def stop_camera(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.is_running = False
            print("Camera stopped.")

    def verify_face(self, face_img):
        try:
            # Try matching with person A
            result_a = DeepFace.verify(face_img, self.database_a + "/image.jpg", enforce_detection=False)
            if result_a['verified']:
                return "Person A"
                
            # Try matching with person B
            result_b = DeepFace.verify(face_img, self.database_b + "/image.jpg", enforce_detection=False)
            if result_b['verified']:
                return "Person B"
                
            return "Unknown"
        except Exception as e:
            print(f"Verification error: {e}")
            return "Unknown"

    def process_frame(self, frame):
        if not self.is_running:
            return frame, "Detector not ready"

        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                # Extract face region
                face_img = frame[y:y+h, x:x+w]
                
                # Verify face
                person = self.verify_face(face_img)
                
                if person != "Unknown":
                    message = self.check_time_and_deduct(person)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{person}: {message}", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    return frame, message
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            return frame, "No recognized face detected"
            
        except Exception as e:
            print(f"Processing error: {e}")
            return frame, f"Processing Error: {e}"

class FaceDetectorApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.detector = FaceDetector()
        self.update_event = None

    def build(self):
        self.title = 'Face Detection System'
        
        root = BoxLayout(orientation='vertical')
        
        # Camera View
        self.camera_widget = Image(allow_stretch=True, keep_ratio=False)
        root.add_widget(self.camera_widget)
        
        # Status Label
        self.status_label = Label(
            text="Press START to begin detection",
            size_hint_y=None,
            height='48dp',
            font_size='20sp'
        )
        root.add_widget(self.status_label)
        
        # Control Button
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

                # Convert frame for Kivy
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_flipped = cv2.flip(frame_rgb, 0)
                
                buf = frame_flipped.tobytes()
                texture = Texture.create(
                    size=(frame_rgb.shape[1], frame_rgb.shape[0]),
                    colorfmt='rgb'
                )
                texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
                self.camera_widget.texture = texture
                
        except Exception as e:
            print(f"Camera update error: {e}")
            self.stop_detection(self.control_button)

    def on_stop(self):
        self.detector.stop_camera()
        if self.update_event:
            Clock.unschedule(self.update_event)

if __name__ == "__main__":
    FaceDetectorApp().run()