import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
from flask import Flask, request, render_template

# Define the model (same architecture as training)
class CNNModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the updated model
model_path = "eurosat_cnn_2.pth"  # Updated model file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(num_classes=2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Flask app setup
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file uploaded")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction="No file selected")

        image = Image.open(file).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        # Prediction
        output = model(image)
        _, predicted = torch.max(output, 1)
        class_names = ["Empty Land", "Constructed Building"]
        prediction = class_names[predicted.item()]

        return render_template('index.html', prediction=prediction, uploaded_image=file.filename)
    
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
