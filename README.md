# digit-recognition-mnist

MNIST Digit Prediction Web App
This project is a web application that allows users to draw a digit (0-9) on a canvas, and a Convolutional Neural Network (CNN) model predicts the drawn digit. The model is trained on the MNIST dataset and deployed using Flask.

# Technologies Used

Python: For backend logic and model training.
PyTorch: For building and training the CNN model.
Flask: For deploying the web application.
HTML/CSS/JavaScript: For the frontend interface.
MNIST Dataset: For training the model.

# Setup Instructions

Prerequisites
Python 3.x: Install Python from python.org.
pip: Python package manager (usually comes with Python).

- Step 1: Clone the Repository
  git clone https://github.com/myehya1/mnist-digit-prediction-web-app.git
  cd digit-recognition-mnist
- Step 2: Install Dependencies
  Install the required Python packages:
  pip install -r requirements.txt
- Step 3: Download the Pre-trained Model
  Download the pre-trained model (mnist_model.pth) and place it in the project root directory. If you want to train the model yourself, follow the instructions in the Training the Model section below.
- Step 4: Run the Flask App
  Start the Flask development server:
  python app.py
  The app will be available at http://127.0.0.1:5000/.

# Training the Model

If you want to train the model yourself, follow these steps:

- Install PyTorch:
  pip install torch torchvision
- Run the Training Script:
  Use the provided Jupyter Notebook (cnn.ipynb) to train the CNN model:
  - Open the notebook in Jupyter or any compatible environment.
  - Run all cells to train the model.
  - The trained model will be saved as mnist_model.pth.
- Replace the Pre-trained Model:
  Replace the existing mnist_model.pth file with the newly trained model.

# Project Structure

mnist-digit-prediction-web-app/
├── app.py # Flask application
├── cnn.ipynb # Jupyter Notebook to train the CNN model
├── basic_nn.ipynb # Jupyter Notebook to train a basic nn model
├── mnist_model.pth # Pre-trained CNN model
├── requirements.txt # Python dependencies
├── static/
│ └── style.css # CSS for the frontend
├── templates/
│ └── index.html # HTML for the frontend
└── README.md # Project documentation

# Usage

Open the web app in your browser: http://127.0.0.1:5000/.
Draw a digit (0-9) on the canvas using your mouse.
Click Predict to see the model's prediction.
Click Clear to reset the canvas and draw a new digit.
