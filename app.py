from flask import Flask, request, jsonify, render_template,send_file,make_response
from flask_cors import CORS
import base64

#Model imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import cv2
#import tflite
#import keras

app = Flask(__name__)
CORS(app)

@app.route('/biopsy',methods=['POST'])
def predict_biopsy():
    '''
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
        #img = cv2.resize(img, (shape, shape))
        #imgs.append(img)
        #model = keras.models.load_model('model/keras_model.tflite')
        #preds = model.predict(np.array(imgs))
        #plt.imsave('predicted.png',preds[15][:,:,0])
        return send_file('predicted.png')
    '''
    #print('form = ',len(request.form), '  |  files = ',len(request.files) )
    print('form = ',request.form['test'], '  |  files = ',request.files)
    f = request.files['file']
    f.save('test.png')
    with open("test.png", "rb") as f:
        image_binary = f.read()

    response = make_response(base64.b64encode(image_binary))
    response.headers.set('Content-Type', 'image/png')
    response.headers.set('Content-Disposition', 'attachment', filename='mask.png')
    return response
    
@app.route('/mammogram',methods=['POST'])
def predict_mammogram():
    #print('form = ',request.data, '  |  files = ',request.get_json())
    data = request.get_json()
    return jsonify({"assess_1": 0.522,
                    "assess_2": 0.142,
                    "assess_3": 0,
                    "assess_4": 0.12,
                    "assess_5": 0.32,
                    "assess_6": 0.1,
    })

# A welcome message to test our server
@app.route('/')
def index():
    #return "<center><h1>Welcome to our Breast Cancer Assistant API !!</h1></center>"
    return render_template('index.html')

#@app.before_first_request
#def init_model():

if __name__ == '__main__':
    app.run(debug=True)
