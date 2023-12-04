import cv2
import torch
import time
import pyttsx3
import numpy as np
import mediapipe as mp
from models import ASLCNN
from PIL import Image
from data import transform

# Class labels for the ASL signs
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space']

cv2.namedWindow('Hand Detection and Classification', cv2.WINDOW_NORMAL)
cv2.namedWindow('Cropped Hand Region', cv2.WINDOW_NORMAL)
frame_count = 0

# Initialize text-to-speech engine
engine = pyttsx3.init()

def crop_hand_region(frame, hand_landmarks, target_size=(200, 200), padding=20):
    try:
        # Extract the bounding box from the hand landmarks
        points = np.array([(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in hand_landmarks.landmark])
        bBox = cv2.boundingRect(points)

        if bBox[2] > 0 and bBox[3] > 0:
            x, y, w, h = bBox
            x -= padding
            y -= padding
            w += 2 * padding
            h += 2 * padding

            # Crop and resize the hand region
            hand_region = frame[max(0, y):min(frame.shape[0], y + h), max(0, x):min(frame.shape[1], x + w)]
            hand_region = cv2.resize(hand_region, target_size)

            return hand_region
        else:
            return None
    except Exception as e:
        print(f"Error in crop_hand_region: {e}")
        return None


def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()


def get_ground_truth_label(frame_count):
    return class_labels[frame_count % len(class_labels)]


def evaluate_accuracy(detected_sign, ground_truth_label, confidence, correct_predictions, total_predictions, confidence_threshold=0.7):
    # Check if the confidence is above the threshold
    if confidence > confidence_threshold:
        total_predictions += 1
        if detected_sign == ground_truth_label:
            correct_predictions += 1

    return correct_predictions, total_predictions

def main():
    # Initialize MediaPipe Hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Load the ASL classification model created in models.py
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ASLCNN(len(class_labels)).to(device)
    model_path = 'data/asl_classifier_state_dict.pth'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Capture video from webcam
    cap = cv2.VideoCapture(0)

    # Hand region target size = 200 x 200 pixels
    target_size = (200, 200)
    correct_predictions = 0
    total_predictions = 0
    total_confidence = 0

    global frame_count
    detected_sign = ""
    last_detection_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(rgb_frame)

        # Check if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Capture an image every 20 frames
                if frame_count % 20 == 0:
                    hand_region = crop_hand_region(frame, hand_landmarks, target_size=target_size, padding=50)

                    if hand_region is not None:
                        cv2.imshow('Cropped Hand Region', hand_region)

                        hand_pil = Image.fromarray(cv2.cvtColor(hand_region, cv2.COLOR_BGR2RGB))
                        hand_region_tensor = transform(hand_pil)

                        # Forward pass through the ASL classification model
                        with torch.no_grad():
                            out = model(hand_region_tensor.unsqueeze(0).to(device))

                        # Get the predicted class
                        _, pred = torch.max(out.data, 1)
                        confidence = torch.nn.functional.softmax(out, dim=1)[0][pred.item()].item()

                        detected_sign = class_labels[pred.item()]
                        print("Detected ASL Sign:", detected_sign)

                        # Update the last detection time
                        last_detection_time = time.time()

                        # Speak out the detected sign
                        text_to_speech(detected_sign)

                        ground_truth_label = get_ground_truth_label(frame_count)
                        correct_predictions, total_predictions = evaluate_accuracy(detected_sign, ground_truth_label,
                                                                                   confidence, correct_predictions,
                                                                                   total_predictions)

                        # Draw the detected ASL sign on the frame if it's within the 3-second window
                        if time.time() - last_detection_time <= 3:
                            cv2.putText(frame, f"ASL Sign: {detected_sign}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 255, 0), 2,
                                        cv2.LINE_AA)

                        # Display the original frame with landmarks
                        cv2.imshow('Hand Detection and Classification', frame)


        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            # Reset frame_count and last_detection_time
            frame_count = 0
            last_detection_time = time.time()
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
