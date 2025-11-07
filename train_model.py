import os
import cv2
import numpy as np
from tqdm import tqdm
from deepface import DeepFace
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import VGG16

def train_model():
    # Define paths and parameters
    database_dir = "database"
    model_output_dir = "models"
    target_size = (224, 224)
    
    # Create models directory if it doesn't exist
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    
    # Collect all images from both classes
    dataset = {
        "a": [],
        "b": []
    }
    
    # Load images and preprocess
    print("Loading and preprocessing images...")
    for person in dataset.keys():
        person_dir = os.path.join(database_dir, person)
        if os.path.exists(person_dir):
            for img_name in tqdm(os.listdir(person_dir)):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(person_dir, img_name)
                    try:
                        # Extract face using DeepFace
                        face_obj = DeepFace.extract_faces(
                            img_path=img_path,
                            detector_backend='opencv',
                            enforce_detection=False
                        )
                        
                        if face_obj and len(face_obj) > 0:
                            face = face_obj[0]['face']
                            # Resize to target size
                            face = cv2.resize(face, target_size)
                            # Normalize pixel values
                            face = face.astype('float32')
                            face /= 255.0
                            dataset[person].append(face)
                            
                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")
                        continue
    
    # Convert to numpy arrays
    X = []  # Images
    y = []  # Labels
    
    print("\nPreparing training data...")
    for idx, (person, images) in enumerate(dataset.items()):
        print(f"Class {person}: {len(images)} images")
        for img in images:
            X.append(img)
            y.append(idx)
    
    X = np.array(X)
    y = np.array(y)
    
    if len(X) == 0:
        raise ValueError("No valid face images found in the database!")
    
    print(f"Final dataset shape: {X.shape}")
    
    # Initialize and train model
    print("Building model...")
    
    # Create base VGG16 model
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(len(dataset), activation='softmax')(x)
    
    # Create final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Training model...")
    history = model.fit(
        X, y,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Save the model
    model_path = os.path.join(model_output_dir, "face_recognition_model.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save labels mapping
    import json
    labels = {str(i): label for i, label in enumerate(dataset.keys())}
    with open(os.path.join(model_output_dir, "labels.json"), "w") as f:
        json.dump(labels, f)

if __name__ == "__main__":
    train_model()