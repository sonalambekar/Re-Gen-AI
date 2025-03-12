import os
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, render_template
from io import BytesIO

app = Flask(__name__)

# Load the trained CNN model
class LandClassifier(nn.Module):
    def __init__(self):
        super(LandClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

model = LandClassifier()
model.load_state_dict(torch.load("eurosat_cnn_2.pth", map_location=torch.device('cpu')))
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Google Maps API Key (replace with your own)
GOOGLE_MAPS_API_KEY = "YOUR_GOOGLE_MAPS_API_KEY"

def get_satellite_image(location):
    """Fetches a satellite image of the specified location from Google Maps API."""
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={location}&zoom=18&size=256x256&maptype=satellite&key={GOOGLE_MAPS_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        location = request.form["location"]
        image = get_satellite_image(location)

        if image is None:
            return render_template("indexMap.html", error="Invalid location. Try again.")

        # Preprocess the image
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Predict
        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1).item()

        result = "Constructed Building" if prediction == 1 else "Empty Land"
        return render_template("indexMap.html", location=location, result=result)
    
    return render_template("indexMap.html")

if __name__ == "__main__":
    app.run(debug=True)
