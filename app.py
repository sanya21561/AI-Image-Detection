from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


app = Flask(__name__)
model = load_model('model.h5')

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((64, 64))  # adjust size as needed
    img = np.expand_dims(img, axis=0)
    return img / 255.0

@app.route('/predict', methods=['POST'])
def predict():
    data = {'success': False}

    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            image_path = 'temp_image.jpg'
            image.save(image_path)

            processed_image = preprocess_image(image_path)
            prediction = model.predict(processed_image)

            data['prediction'] = float(prediction[0][0])
            data['success'] = True

    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
