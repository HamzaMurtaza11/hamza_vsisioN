from flask import Flask, request, jsonify
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)

# Assuming the model is relatively large and causing timeouts due to loading times or inference processing,
# ensure it is loaded efficiently and consider using a lighter model if possible.
model = load_model("keras_model_hamza_lite.h5", compile=False)
class_names = [line.strip() for line in open("labels_lite.txt", "r")]

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "file is required"}), 400
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Streamlining the image preprocessing
        with Image.open(file).convert("RGB") as image:
            image = ImageOps.fit(image, (224, 224), Image.ANTIALIAS)
            image_array = np.asarray(image) / 127.5 - 1  # Combined steps to reduce memory footprint

            prediction = model.predict(np.expand_dims(image_array, axis=0))
            index = np.argmax(prediction)
            return jsonify({
                "class": class_names[index],
                "confidence_score": float(prediction[0][index])
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500




    



