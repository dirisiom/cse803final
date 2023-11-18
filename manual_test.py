import cv2
import torch
from PIL import Image
from torchvision import transforms

from models import ASLCNN
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'SPACE', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ASLCNN(len(class_labels)).to(device)
model_path = 'data/asl_classifier_state_dict.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


# Function to preprocess and transform the image
def preprocess_and_transform_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found")

    # Convert BGR to RGB (as OpenCV loads in BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to PIL format for applying transforms
    image_pil = Image.fromarray(image_rgb)
    cropped = image_pil.crop((5,5,195,195))

    # Apply the transformations
    image_tensor = transform(cropped)

    return image_tensor, cropped


# Function to predict the class of an image
def classify_image(image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension and send to device
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        return class_labels[predicted.item()]


# Replace 'path_to_your_image.jpg' with the path to the image you want to classify
image_path = './data/asl_images/SPACE/space/space2.jpg'
image_tensor, transformed_image_pil = preprocess_and_transform_image(image_path)
predicted_class = classify_image(image_tensor)
print(predicted_class)

# Display the preprocessed image
plt.imshow(transformed_image_pil)
plt.title(f"Predicted Class: {predicted_class}")
plt.show()