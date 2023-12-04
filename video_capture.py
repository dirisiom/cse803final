import cv2
import torch
import time
import pyttsx3
import numpy as np
import mediapipe as mp
from models import ASLCNN
from PIL import Image, ImageTk
import tkinter as tk
from data import transform as data_transform

# Class labels for the ASL signs
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z', 'space']


class ASLApp:
    def __init__(self, master):
        self.master = master
        self.master.title("ASL Sign Detection and Classification")
        self.detected_sign = tk.StringVar()
        self.detected_sign.set("")

        # Create labels
        self.label_current_sign = tk.Label(master, text="Current ASL Sign:", font=('Helvetica', 14), padx=10, pady=10)
        self.label_current_sign.grid(row=0, column=0)
        self.label_detected_sign = tk.Label(master, textvariable=self.detected_sign, font=('Helvetica', 14), padx=10,
                                            pady=10)
        self.label_detected_sign.grid(row=0, column=1)

        # Create Speak button
        self.btn_speak = tk.Button(master, text="Speak", command=self.speak_current_sign, font=('Helvetica', 12),
                                   padx=10, pady=10)
        self.btn_speak.grid(row=0, column=2, pady=20)  # Moved to column 2

        # Create canvas for camera feed
        self.canvas = tk.Canvas(master, width=640, height=480)
        self.canvas.grid(row=1, column=0, columnspan=3)

        # Create a StringVar for detected sign
        self.detected_sign = tk.StringVar()
        self.detected_sign.set("")

        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()

        # Initialize MediaPipe Hands module and ASL classification model
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ASLCNN(len(class_labels)).to(self.device)
        model_path = 'data/asl_classifier_state_dict.pth'
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

        self.cap = cv2.VideoCapture(0)

        # Hand region target size = 200 x 200 pixels
        self.target_size = (200, 200)
        self.frame_count = 0
        self.detected_sign.set("No sign detected")

        # Start the update loop
        self.update()

    def speak_current_sign(self):
        detected_sign = self.detected_sign.get()
        if detected_sign:
            self.text_to_speech(detected_sign)

    def crop_hand_region(self, frame, hand_landmarks, target_size=(200, 200), padding=20):
        try:
            # Extract the bounding box from the hand landmarks
            points = np.array(
                [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in hand_landmarks.landmark])
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

    def text_to_speech(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Hands
            results = self.hands.process(rgb_frame)

            # Check if hands are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Capture an image every 20 frames
                    if self.frame_count % 20 == 0:
                        hand_region = self.crop_hand_region(frame, hand_landmarks, target_size=self.target_size,
                                                            padding=100)

                        if hand_region is not None:
                            hand_pil = Image.fromarray(cv2.cvtColor(hand_region, cv2.COLOR_BGR2RGB))
                            hand_region_tensor = data_transform(hand_pil).unsqueeze(0).to(self.device)

                            # Forward pass through the ASL classification model
                            with torch.no_grad():
                                out = self.model(hand_region_tensor)

                            # Get the predicted class
                            _, pred = torch.max(out.data, 1)
                            detected_sign = class_labels[pred.item()]

                            # Update the last detection time
                            self.last_detection_time = time.time()

                            # Set the detected sign
                            self.detected_sign.set(detected_sign)

                            # Draw the detected ASL sign on the frame if it's within the 3-second window
                            if time.time() - self.last_detection_time <= 3:
                                self.label_current_sign.config(text=f"Current ASL Sign: {detected_sign}")

            # Display the original frame with landmarks on the canvas
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(img)

            # Ensure that the image reference is retained
            self.canvas.img_tk = img_tk

            # Update the canvas image
            self.canvas.config(width=img_tk.width(), height=img_tk.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

            # Update every 10 milliseconds
            self.master.after(10, self.update)

        else:
            # Release the camera when the window is closed
            self.cap.release()


# Create the main window
root = tk.Tk()
app = ASLApp(root)
root.mainloop()