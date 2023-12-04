import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import torch
import os

from PIL import Image
from torchvision import transforms

from models import ASLCNN

transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z', 'space']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ASLCNN(len(class_labels)).to(device)
model_path = 'data/asl_classifier_state_dict.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


def crop_hand_region(frame, hand_landmarks, target_size=(200, 200), padding=100):
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


# Function to preprocess and transform the image
def preprocess_and_transform_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found")

    # Convert BGR to RGB (as OpenCV loads in BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.imshow(image_rgb)
    # plt.show()

    hands = mp.solutions.hands.Hands()
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        region = crop_hand_region(image_rgb, results.multi_hand_landmarks[0])
        image_pil = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))

    # Apply the transformations
    image_tensor = transform(image_pil)

    return image_tensor, image_pil


# Function to predict the class of an image
def classify_image(i_t):
    with torch.no_grad():
        i_t = i_t.unsqueeze(0).to(device)  # Add batch dimension and send to device
        output = model(i_t)
        _, predicted = torch.max(output, 1)
        return class_labels[predicted.item()]

dpath = './data/RealWorldData'
correct = 0
total = 0
for dir in os.listdir(dpath):
    label = dir
    for im in os.listdir(f'{dpath}/{dir}'):
        total += 1
        image_path = F"{dpath}/{dir}/{im}"
        image_tensor, transformed_image_pil = preprocess_and_transform_image(image_path)
        predicted_class = classify_image(image_tensor)
        print(predicted_class)
        if predicted_class == label:
            correct += 1
        # if label == 'A':
        #     # Display the preprocessed image
        #     plt.imshow(transformed_image_pil)
        #     plt.title(f"Predicted Class: {predicted_class}")
        #     plt.show()

print(f"{correct} correct out of {total}")


