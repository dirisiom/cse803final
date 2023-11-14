import cv2
import torch
import numpy as np
import mediapipe as mp
from models import ASLCNN

# Class labels for the ASL signs
class_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "SPACE", "DELETE", "NOTHING"]


def crop_hand_region(frame, hand_landmarks, target_size=(200, 200)):
    try:
        points = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
        x, y, w, h = cv2.boundingRect(np.float32(points))
        hand_region = frame[y:y + h, x:x + w]

        # Resize and normalize the hand region
        hand_region = cv2.resize(hand_region, target_size)
        hand_region = hand_region.astype(np.float32) / 255.0

        return hand_region
    except Exception as e:
        print(f"Error in crop_hand_region: {e}")
        return None


def main():
    # Initialize MediaPipe Hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Load the ASL classification model created in models.py
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ASLCNN(29).to(device)
    model_path = 'data/asl_classifier_state_dict.pth'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Capture video from webcam
    cap = cv2.VideoCapture(0)

    # Hand region target size = 200 x 200 pixels
    target_size = (200, 200)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB for MediaPipe and process the frame with MediaPipe Hands
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Check if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on hand
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Crop the region around the hand and resize
                hand_region = crop_hand_region(frame, hand_landmarks, target_size=target_size)

                if hand_region is not None:
                    # Convert hand region to a PyTorch tensor
                    hand_region = torch.from_numpy(hand_region.transpose((2, 0, 1))).float()

                    # Forward pass through the ASL classification model
                    with torch.no_grad():
                        out = model(hand_region.unsqueeze(0).to(device))

                    # Get the predicted class
                    _, pred = torch.max(out.data, 1)
                    print("Detected ASL Sign:", class_labels[pred.item()])

        cv2.imshow('Hand Detection and Classification', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    # Close the windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
