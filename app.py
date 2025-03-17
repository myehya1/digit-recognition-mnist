from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os

app = Flask(__name__)

# Define the CNN model (same as your training code)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)  # Flatten
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Load the trained model
model = CNN()
model.load_state_dict(torch.load('mnist_model.pth', map_location=torch.device('cpu')))
model.eval()

# Preprocess the image
def preprocess_image(image):
    # Resize to 28x28 and convert to grayscale
    image = image.resize((28, 28)).convert('L')
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    # Read the image file
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Preprocess and predict
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        prediction = predicted.item()

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 for local development
    app.run(host='0.0.0.0', port=port)