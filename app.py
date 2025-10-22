import os
import sys
import torch
import torch.nn as nn
from torchvision import models
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import torchvision.transforms as transforms
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_metal_defect_model.pth')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Simple CNN model as fallback
class SimpleMetalDefectCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(SimpleMetalDefectCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def load_trained_model(model_path):
    """Load the trained model with enhanced error handling"""
    try:
        logger.info("🔄 Loading trained model...")
        
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found: {model_path}")
            return None, None, None
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Get model configuration
        class_names = checkpoint.get('class_names', ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches'])
        num_classes = len(class_names)
        
        # Determine device - force CPU for compatibility
        device = torch.device("cpu")
        logger.info(f"Using device: {device}")
        
        # Try different model architectures
        model = None
        model_error = ""
        
        try:
            # Try ResNet50 first
            model = models.resnet50(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, 1024),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(1024),
                nn.Dropout(0.4),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
            logger.info("✅ Using ResNet50 architecture")
        except Exception as e1:
            model_error = str(e1)
            try:
                # Fallback to simple CNN
                model = SimpleMetalDefectCNN(num_classes=num_classes)
                logger.info("✅ Using Simple CNN architecture")
            except Exception as e2:
                logger.error(f"❌ Both model architectures failed: {e1}, {e2}")
                return None, None, None
        
        if model is None:
            return None, None, None
            
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"📊 Model accuracy: {checkpoint.get('accuracy', 'Unknown')}")
        logger.info(f"🎯 Number of classes: {num_classes}")
        logger.info(f"📝 Class names: {class_names}")
        
        return model, class_names, device
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return None, None, None

# Load model at startup
logger.info("Starting model loading...")
model, class_names, device = load_trained_model(MODEL_PATH)

# Define transforms for prediction
prediction_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image):
    """
    Predict the class of an image
    """
    try:
        if model is None:
            return {'success': False, 'error': 'Model not loaded'}
        
        # Transform and predict
        image_tensor = prediction_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        predicted_label = class_names[predicted_class.item()]
        confidence_score = confidence.item()
        all_probabilities = probabilities.cpu().numpy()[0]
        
        # Create probability dictionary
        prob_dict = {}
        for i, class_name in enumerate(class_names):
            prob_dict[class_name] = float(all_probabilities[i])
        
        return {
            'predicted_class': predicted_label,
            'confidence': float(confidence_score),
            'all_probabilities': prob_dict,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"❌ Error during prediction: {e}")
        return {
            'success': False,
            'error': str(e)
        }

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for predicting defect type from uploaded image
    """
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if file and allowed_file(file.filename):
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            result = predict_image(image)
            
            if result['success']:
                return jsonify({
                    'success': True,
                    'predicted_class': result['predicted_class'],
                    'confidence': result['confidence'],
                    'all_probabilities': result['all_probabilities']
                })
            else:
                return jsonify({
                    'success': False,
                    'error': result['error']
                }), 500
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Allowed types: png, jpg, jpeg, bmp, gif'
            }), 400
            
    except Exception as e:
        logger.error(f"❌ Error in predict endpoint: {e}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device),
        'classes': class_names if class_names else []
    })

if __name__ == '__main__':
    if model is None:
        logger.error("❌ WARNING: Model failed to load. API will start but predictions will fail.")
    else:
        logger.info("✅ Model loaded successfully. Starting Flask server...")
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)