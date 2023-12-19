import flask
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import base64
import io

app = Flask(__name__, template_folder='templates')

# Load the pre-trained model in the global scope
model = load_model('HAM-REsnet50.h5')

# Map model class indices to human-readable class names
class_names = [ 'Benign keratosis like lesions','Melanocytic nevi', 'Dermatofibroma', 'Melanoma','Vascular lesions', 'Basal cell carcinoma', 'Actinic keratoses' ]
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from the form
        image_data = request.form['image_data'].split(",")[1]  # Fix index to get the base64 data

        # Convert base64 image data to NumPy array
        img = Image.open(io.BytesIO(base64.b64decode(image_data)))

        # Check image format and raise an error if it's not supported
        if img.format not in ['JPEG', 'PNG']:
            raise ValueError('Invalid image format. Supported formats: JPEG, PNG')

        # Resize the image to the input size expected by the model
        img = img.resize((70, 70))

        # Convert the image to a NumPy array
        img_array = tf.keras.preprocessing.image.img_to_array(img)

        # Expand dimensions to match the input shape expected by the model
        img_array = np.expand_dims(img_array, axis=0)  # Fix axis

        # Make predictions using the loaded model
        predictions = model.predict(img_array)

        # Get the predicted class index
        predicted_class_index = np.argmax(predictions)

        # Check if the predicted class index is within the valid range
        if 0 <= predicted_class_index < len(class_names):
            # Map the class index to human-readable class name
            predicted_class = class_names[predicted_class_index]
            confidence = predictions[0][predicted_class_index]  # Adjust index
            return render_template('result.html', predicted_class=predicted_class, confidence=confidence)
        else:
            # Handle the case when the predicted class index is out of range
            return render_template('result_unknown.html')

    except ValueError as ve:
        # Handle specific errors (e.g., invalid image format)
        return jsonify({'error': str(ve)})

    except Exception as e:
        # Handle other exceptions and return a meaningful error message
        return jsonify({'error': str(e)})

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/start')
def start():
    return render_template('start.html')

if __name__ == '__main__':
    app.run(debug=True)




# from flask import Flask, render_template, request, jsonify, g
# from tensorflow.keras.models import load_model
# from PIL import Image
# import numpy as np
# import base64
# import io

# app = Flask(__name__, template_folder='templates')

# # Load the pre-trained model only once during application startup
# def load_model():
#     return load_model('HAM.h5')

# # Use Flask's application context to store the model
# def get_model():
#     if 'model' not in g:
#         g.model = load_model()
#     return g.model

# # Map model class indices to human-readable class names
# class_names = ['melanoma', 'melanocytic nevi','basal cell carcinoma', 'benign keartosis', 'actinic keratosis', 'dermatofibroma', 'vascular lesions' ]

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get image data from the form
#         image_data = request.form['image_data'].split(",")[1]

#         # Convert base64 image data to NumPy array
#         img = Image.open(io.BytesIO(base64.b64decode(image_data)))
#         img = img.resize((28, 28))  # Adjust size as needed
#         img_array = tf.keras.preprocessing.image.img_to_array(img)

#         # Preprocess the image data
#         img_array = np.expand_dims(img_array, axis=0)

#         # Make predictions using the loaded model
#         model = get_model()
#         predictions = model.predict(img_array)

#         # Get the predicted class index
#         predicted_class_index = np.argmax(predictions)

#         # Check if the predicted class index is within the valid range
#         if 0 <= predicted_class_index < len(class_names):
#             # Map the class index to human-readable class name
#             predicted_class = class_names[predicted_class_index]
#             confidence = predictions[0][predicted_class_index]
#             return render_template('result.html', predicted_class=predicted_class, confidence=confidence)
#         else:
#             # Handle the case when the predicted class index is out of range
#             return render_template('result_unknown.html')

#     except Exception as e:
#         # Log the error for debugging purposes
#         app.logger.error(str(e))
#         # Handle exceptions and return a meaningful error message
#         return jsonify({'error': 'An error occurred during prediction.'})

# @app.route('/')
# def index():
#     return render_template("index.html")

# @app.route('/start')
# def start():
#     return render_template('start.html')

# if __name__ == '__main__':
#     app.run(debug=True)
