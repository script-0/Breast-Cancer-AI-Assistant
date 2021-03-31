from flask import Flask, request, jsonify
from flask_cors import CORS

#Model imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras

app = Flask(__name__)
CORS(app)

@app.route('/biopsy',methods=['POST'])
def predict_biopsy():
    f =  request.files['file']
    f.save('to_predict.png')
    imgs = [] 
    shape = 256
    for i in range(15):
        img = plt.imread('dataset/Dataset_BUSI_with_GT/f{}.png'.format(i+1))
        img = cv2.resize(img, (shape, shape))
        imgs.append(img)
    shape=256
    img = plt.imread('to_predict.png')
    img = cv2.resize(img, (shape, shape))
    imgs.append(img)
    model = keras.models.load_model('model/keras_biopsy.h5')
    preds = model.predict(np.array(imgs))
    plt.imsave('predicted.png',preds[15][:,:,0])
    return send_file('predicted.png')

# A welcome message to test our server
@app.route('/')
def index():
    return "<center><h1>Welcome to our Breast Cancer Assistant API !!</h1></center>"

#@app.before_first_request
#def init_model():

if __name__ == '__main__':
    app.run()
