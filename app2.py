from flask import Flask, request, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

app = Flask(__name__)

# Load trained model
model_path = "eurosat_cnn_2.pth"  # Ensure this file exists in your project directory
class_labels = ["Empty Land", "Constructed Building"]

# Define CNN model structure (must match the saved model)
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

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel(num_classes=2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Home Route
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    suggestions = ""
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index2.html', prediction="No file uploaded")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index2.html', prediction="No file selected")
        
        image = Image.open(file).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image)
            _, predicted_class = torch.max(output, 1)
            prediction = class_labels[predicted_class.item()]
        
        # Provide suggestions if empty land is detected
        if prediction == "Empty Land":
            suggestions = """
            <div class="flex-container">
        <!-- Possible Uses of Empty Land List -->
        <div class="list-container">
            <h3>Possible Uses of Empty Land:</h3>
            <ul>
                        <li>Organic Farming – Grow fruits, vegetables, or herbs without chemicals.</li>
                        <li>Mushroom Farming – Low-maintenance and profitable, even in small spaces.</li>
                        <li>Bird Sanctuary or Butterfly Garden – Attract wildlife and promote conservation.</li>
                        <li>Community Garden – Let locals grow food together, promoting social bonding.</li>
                        <li>Sustainable Co-Living Spaces – Rent out small eco-homes or tiny houses.</li>
                    </ul>
        </div>

        <!-- Biotech Solution List -->
        <div class="list-container">
            <h3>Biotech Solutions:</h3>
            <ul>
                <li>Oil Spill Cleanup – Certain bacteria like Pseudomonas and Alcanivorax break down petroleum-based pollutants.</li>
                        <li>Mycoremediation (Fungi Cleanup) – Mushrooms like Pleurotus ostreatus (Oyster mushrooms) absorb toxins and break down pollutants.</li>
                        <li>Electro-Bioremediation – A combination of electrical currents and bacteria to remove metal contaminants.</li>
                        <li>Biodegradable Bioplastics – Enhancing soil health by replacing plastic waste with compostable bio-based materials.</li>
            </ul>
        </div>
    </div>
            """
    
    return render_template('index2.html', prediction=prediction, suggestions=suggestions)

if __name__ == "__main__":
    app.run(debug=True)