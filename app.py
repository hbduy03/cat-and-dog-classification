from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = load_model('catanddog.h5')
classes = {0: 'cat', 1: 'dog'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))
    img = img.resize((150,150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prob = model.predict(img_array)[0][0]
    label = 1 if prob > 0.5 else 0
    confidence = float(prob) if label==1 else float(1-prob)
    result = f"Prediction: {classes[label].upper()} Confidence: {confidence:.4f}"
    return render_template('index.html',result = result)

if __name__ == '__main__':
    app.run(debug=True)
