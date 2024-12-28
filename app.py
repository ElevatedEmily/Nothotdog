from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)

# Load ONNX model
onnx_model_path = "resnet_model.onnx"
session = ort.InferenceSession(onnx_model_path)

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match the input size
    image = np.array(image).astype('float32')  # Ensure float32 type
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # Apply normalization
    image = np.transpose(image, (2, 0, 1))  # Convert to CHW format
    tensor = np.expand_dims(image, axis=0)  # Add batch dimension
    
    # Debugging: Print the tensor's data type
    print(f"Preprocessed tensor data type: {tensor.dtype}")
    
    return tensor


@app.route("/")
def home():
    return "Welcome to the ResNet Image Classifier API!"
@app.route("/favicon.ico")
def favicon():
    return "", 204  # No content

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    try:
        file = request.files["file"]
        image = Image.open(file.stream).convert("RGB")
        input_tensor = preprocess_image(image).astype('float32')  # Ensure float32 explicitly
        
        # Run inference
        outputs = session.run(None, {"input": input_tensor})
        prediction = outputs[0][0][0]
        result = "Positive" if prediction > 0.5 else "Negative"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
