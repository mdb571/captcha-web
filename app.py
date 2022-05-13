import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
#:from skimage import io
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
import imutils
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import pickle

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()

# You can also use pretrained model from Keras
# Check https://keras.io/applications/

model =tf.keras.models.load_model('captcha_model.hdf5',compile=False)
print('Model loaded. Check http://127.0.0.1:5000/')


with open("model_labels.dat", "rb") as f:
    lb = pickle.load(f)

def resize_to_fit(image, width, height):
    
    (h, w) = image.shape[:2]

    if w > h:
        image = imutils.resize(image, width=width)

    else:
        image = imutils.resize(image, height=height)

    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)
    
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
        cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    return image

def model_predict(img_path, model):
    image = cv2.imread(img_path)
    kernel = np.ones((5,5), np.uint8)
    erosion_image = cv2.erode(image, kernel, iterations=1) 
    img = cv2.dilate(image, kernel, iterations=1)
    img = Image.fromarray(img)

    predictions = []

    width, height = img.size
    left = 0
    top = 0
    right = (width-1)/4
    bottom = height - 1
    im1 = img.crop((left,top,right,bottom))
    im1 = np.asarray(im1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

    im1 = resize_to_fit(im1, 20, 20)
   
    im1 = np.expand_dims(im1, axis=2)
    im1 = np.expand_dims(im1, axis=0)

    prediction = model.predict(im1)
    print(prediction)
    letter = lb.inverse_transform(prediction)[0]
    predictions.append(letter)

    ###############################################################

    left = (width-1)/4
    top = 0
    right = (width-1)/6*2
    bottom = height - 1
    im1 = img.crop((left,top,right,bottom))
    im1 = np.asarray(im1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im1 = resize_to_fit(im1, 20, 20)

    im1 = np.expand_dims(im1, axis=2)
    im1 = np.expand_dims(im1, axis=0)

    prediction = model.predict(im1)
    print(prediction)
    letter = lb.inverse_transform(prediction)[0]
    predictions.append(letter)

    ##################################################################

    left = (width-1)/6*2
    top = 0
    right = (width-1)/6*2.7
    bottom = height - 1
    im1 = img.crop((left,top,right,bottom))

    im1 = np.asarray(im1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im1 = resize_to_fit(im1, 20, 20)

    im1 = np.expand_dims(im1, axis=2)
    im1 = np.expand_dims(im1, axis=0)

    prediction = model.predict(im1)
    print(prediction)
    letter = lb.inverse_transform(prediction)[0]
    predictions.append(letter)

    ######################################################################

    left = (width-1)/6*2.7
    top = 0
    right = (width-1)/6*3.3
    bottom = height - 1
    im1 = img.crop((left,top,right,bottom))
    im1 = np.asarray(im1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im1 = resize_to_fit(im1, 20, 20)

    im1 = np.expand_dims(im1, axis=2)
    im1 = np.expand_dims(im1, axis=0)

    prediction = model.predict(im1)
    print(prediction)
    letter = lb.inverse_transform(prediction)[0]
    predictions.append(letter)

    #######################################################################

    left = (width-1)/6*3.3
    top = 0
    right = (width-1)/6*4
    bottom = height - 1
    im1 = img.crop((left,top,right,bottom))
    im1 = np.asarray(im1)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im1 = resize_to_fit(im1, 20, 20)

    im1 = np.expand_dims(im1, axis=2)
    im1 = np.expand_dims(im1, axis=0)

    prediction = model.predict(im1)

    letter = lb.inverse_transform(prediction)[0]
    predictions.append(letter)
    print(prediction)

    captcha_text = "".join(predictions)
    return captcha_text


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print(preds)
        
        return preds
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
    app.run()
