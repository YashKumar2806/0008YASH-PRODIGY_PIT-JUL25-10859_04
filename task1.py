import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import mediapipe as mp

# Constants
DATASET_PATH = "datasets/leapGestRecog"
IMAGE_SIZE = (64, 64)
EPOCHS = 10
MODEL_PATH = "gesture_model.h5"
LABEL_DICT_PATH = "label_dict.pkl"

# Load data function
def load_data(dataset_path):
    images = []
    labels = []
    label_dict = {}
    label_id = 0

    for subject_folder in sorted(os.listdir(dataset_path)):
        subject_path = os.path.join(dataset_path, subject_folder)
        if not os.path.isdir(subject_path):
            continue

        for gesture_folder in sorted(os.listdir(subject_path)):
            gesture_path = os.path.join(subject_path, gesture_folder)
            if not os.path.isdir(gesture_path):
                continue

            if gesture_folder not in label_dict:
                label_dict[gesture_folder] = label_id
                label_id += 1
            label = label_dict[gesture_folder]

            for file in os.listdir(gesture_path):
                img_path = os.path.join(gesture_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, IMAGE_SIZE)
                images.append(img)
                labels.append(label)

    if not images:
        raise ValueError("❌ No images found. Check your dataset path and folder structure!")

    images = np.array(images).reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1) / 255.0
    labels = to_categorical(labels, num_classes=len(label_dict))
    print("Label Mapping:", label_dict)
    return train_test_split(images, labels, test_size=0.2, random_state=42), label_dict

# Build CNN model
def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Real-time prediction with webcam
def live_prediction(model, label_dict):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture()
    hands = mp_hands.Hands()

    labels_rev = {v: k for k, v in label_dict.items()}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                x_min = min([lm.x for lm in hand_landmarks.landmark])
                y_min = min([lm.y for lm in hand_landmarks.landmark])
                x_max = max([lm.x for lm in hand_landmarks.landmark])
                y_max = max([lm.y for lm in hand_landmarks.landmark])

                x1, y1 = int(x_min * w) - 20, int(y_min * h) - 20
                x2, y2 = int(x_max * w) + 20, int(y_max * h) + 20
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                roi_resized = cv2.resize(roi_gray, IMAGE_SIZE)
                roi_input = roi_resized.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1) / 255.0

                prediction = model.predict(roi_input, verbose=0)
                class_id = np.argmax(prediction)
                gesture_name = labels_rev.get(class_id, "Unknown")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, gesture_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Live Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main menu
if __name__ == "__main__":
    print("1. Train model (from Kaggle ASL dataset)")
    print("2. Live webcam prediction")
    choice = input("Choose an option (1/2): ")

    if choice == '1':
        (X_train, X_test, y_train, y_test), label_dict = load_data(DATASET_PATH)
        with open(LABEL_DICT_PATH, 'wb') as f:
            pickle.dump(label_dict, f)

        model = build_model((IMAGE_SIZE[0], IMAGE_SIZE[1], 1), len(label_dict))
        model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_test, y_test))
        model.save(MODEL_PATH)
        print("✅ Model trained and saved.")

    elif choice == '2':
        if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_DICT_PATH):
            print("❌ Model or label dictionary not found. Please train it first.")
        else:
            with open(LABEL_DICT_PATH, 'rb') as f:
                label_dict = pickle.load(f)
            model = load_model(MODEL_PATH)
            live_prediction(model, label_dict)

    else:
        print("❌ Invalid option. Please choose 1 or 2.")
