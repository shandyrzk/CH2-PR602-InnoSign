import os
import keras
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'png', 'jpeg'])
app.config['UPLOAD_FOLDER'] = ''

def allowed_extension(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def getLabel():
    labels = [
    'additional', 'alcohol', 'allergy', 'bacon', 'bag', 'barbecue', 'bill', 'biscuit', 'bitter', 'bread', 'burger', 'bye', 
    'cake', 'cash', 'cheese', 'chicken', 'coke', 'cold', 'cost', 'coupon', 'credit card', 'cup', 'dessert', 'drink', 'drive', 
    'eat', 'eggs', 'enjoy', 'fork', 'french fries', 'hello', 'hot', 'icecream', 'ingredients', 'juicy', 'ketchup', 'lactose', 
    'lettuce', 'lid', 'manager', 'menu', 'milk', 'mustard', 'napkin', 'no', 'order', 'pepper', 'pickle', 'pizza', 'please', 
    'ready', 'receipt', 'refill', 'repeat', 'safe', 'salt', 'sandwich', 'sauce', 'small', 'soda', 'sorry', 'spicy', 'spoon', 
    'straw', 'sugar', 'sweet', 'thank-you', 'tissues', 'tomato', 'total', 'urgent', 'vegetables', 'wait', 'warm', 'water', 
    'what', 'would', 'yoghurt', 'your'
    ]
    return labels

# Load pre-trained model
def loadmodel():
    model = keras.models.load_model('model.h5')
    return model

def predict_class(image_path):
    model = loadmodel()
    # Load and preprocess the image
    img = Image.open(image_path)
    img = img.resize((128, 128))
    img_array = np.asarray(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Get model predictions
    predictions = model.predict(img_array)[0]
    likely_class = np.argmax(predictions)

    return likely_class

@app.route("/", methods=['GET'])
def homepage():
    return jsonify({
        "data": None,
        "status": {
            "code": 200,
            "message": "API is running"
        },
    }), 200

@app.route("/api/predict", methods=['POST'])
def prediction():
    if request.method == 'POST':
        image = request.files["file"]
        if image and allowed_extension(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            class_names = getLabel()
            predicted_class = predict_class(image_path)
            predicted_class_name = class_names[predicted_class]
            # print(predicted_class)

            os.remove(image_path)

            return jsonify({
                "data": {
                    "class_name": predicted_class_name ,
                },
                "status": {
                    "code": 200,
                    "message": "Success predicting image"
                },
            }), 200
        else:
            return jsonify({
                "data": None,
                "status": {
                    "code": 400,
                    "message": "Invalid image extension. Only accept jpg, jpeg, and png."
                },
            }), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
