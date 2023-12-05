import time
import tkinter as tk
from tkinter import scrolledtext

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import torch
from PIL import Image, ImageTk

from data import transform
from models import ASLCNN

# Class labels for the ASL signs
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z', 'space']

def data_transform(image):
    return transform(image)

class ASLApp:
    def __init__(self, master):
        self.master = master
        self.master.title("ASL Sign Detection and Classification")
        self.detected_sign = tk.StringVar()
        self.detected_sign.set("")
        self.is_space_pressed = False
        self.sentence = ""

        # Create labels
        self.label_current_sign = tk.Label(master, text="Current ASL Sign:", font=('Helvetica', 14), padx=10, pady=10)
        self.label_current_sign.grid(row=0, column=0)

        # Create label for generated sentence
        self.label_generated_sentence = tk.Label(master, text="Generated Sentence:", font=('Helvetica', 14), padx=10,
                                                 pady=10)
        self.label_generated_sentence.grid(row=0, column=2)

        # Create Speak button
        self.btn_speak = tk.Button(master, text="Speak", command=self.speak_generated_sentence, font=('Helvetica', 12),
                                   padx=10, pady=10, bg='blue', fg='white')  # Set background color to blue
        self.btn_speak.grid(row=0, column=3)

        # Create scrolled text box for the sentence
        self.sentence_text = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=40, height=10)
        self.sentence_text.grid(row=1, column=0, columnspan=4, pady=10, padx=(10, 20), sticky=tk.E)

        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()

        # Initialize MediaPipe Hands module and ASL classification model
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ASLCNN(len(class_labels)).to(self.device)
        model_path = 'data/asl_classifier_state_dict_great.pth'
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

        self.cap = cv2.VideoCapture(0)

        # Hand region target size = 200 x 200 pixels
        self.target_size = (200, 200)
        self.frame_count = 0
        self.detected_sign.set("No sign detected")

        # Bind the space bar to the space_pressed method
        self.master.bind("<space>", self.space_pressed)

        # Create canvas for camera feed
        self.canvas = tk.Canvas(master, width=640, height=480)
        self.canvas.grid(row=2, column=0, columnspan=4)

        # Start the update loop
        self.update()

    def speak_generated_sentence(self):
        sentence = self.sentence_text.get("1.0", tk.END).strip()
        if sentence:
            self.sentence = sentence
            self.text_to_speech()
            self.sentence_text.delete("1.0", tk.END)

    def space_pressed(self, event):
        if self.detected_sign.get():
            # Replace 'space' with actual space
            sign = self.detected_sign.get() if self.detected_sign.get() != 'space' else ' '
            self.sentence_text.insert(tk.END, sign)
            self.is_space_pressed = True

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

    def text_to_speech(self):
        self.engine.say(self.sentence)
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
                                                            padding=50)

                        if hand_region is not None:
                            hand_pil = Image.fromarray(cv2.cvtColor(hand_region, cv2.COLOR_BGR2RGB))
                            # plt.imshow(hand_pil)
                            # plt.show()
                            hand_region_tensor = data_transform(hand_pil).unsqueeze(0).to(self.device)

                            # Forward pass through the ASL classification model
                            with torch.no_grad():
                                out = self.model(hand_region_tensor)

                            # Get the predicted class
                            _, pred = torch.max(out.data, 1)
                            detected_sign = class_labels[pred.item()]
                            print(detected_sign)

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

            self.canvas.img_tk = img_tk
            self.canvas.config(width=img_tk.width(), height=img_tk.height())
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
  
            if self.is_space_pressed:
                self.is_space_pressed = False
                self.detected_sign.set("")

            self.master.after(10, self.update)

        else:
            # Release the camera when the window is closed
            self.cap.release()

# Create the main window
root = tk.Tk()
app = ASLApp(root)
root.mainloop()
