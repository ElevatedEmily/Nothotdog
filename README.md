# Nothotdog
Hotdog not hotdog ResNet from Silicon Valley Season 4. 
This is a ResNet-50 model trained on a kaggle dataset found here: https://www.kaggle.com/datasets/thedatasith/hotdog-nothotdog\
The website is made using Flask and react. 

# Set up
    git clone https://github.com/ElevatedEmily/Nothotdog.git
    cd hotdog-nothotdog
    npx create-react-app .
    npm install @mui/material @mui/icons-material axios react-router-dom
    pip install torch torchvision matplotlib scikit-learn optuna tqdm pillow flask flask-cors onnxruntime

# Training the model
ensure the kaggle dataset is in the project directory

    python CNN.py
    python Covert2ONNX.pu

# Run Backend and Frontend

    python app.py
    npm start
Have Fun!
