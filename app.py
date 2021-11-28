from flask import Flask, request, jsonify, send_file , make_response, render_template
from flask_cors import CORS
import base64

#Model imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
from joblib import load

app = Flask(__name__)
CORS(app)

#Global variables
global min_age
global max_age 
global max_bmi
global min_bmi
global final_df_columns
global model

#global seed
seed=11

#global DATA_PATH
DATA_PATH = "dataset/breast_mammogram_dataset.csv"

#global max_age
max_age=0

#global min_age
min_age=0 

#global max_bmi
max_bmi=0

#global min_bmi
min_bmi=0

#global final_df_columns
final_df_columns = ['age_c', 
                    'bmi_c', 
                    'density_c_1.0', 
                    'density_c_2.0', 
                    'density_c_3.0',
                    'density_c_4.0', 
                    'famhx_c_0.0', 
                    'famhx_c_1.0', 
                    'famhx_c_9.0',
                    'hrt_c_0.0', 
                    'hrt_c_1.0', 
                    'hrt_c_9.0',
                    'prvmam_c_0.0', 
                    'prvmam_c_1.0',
                    'prvmam_c_9.0',
                    'biophx_c_0.0', 
                    'biophx_c_1.0', 
                    'biophx_c_9.0',
                    'mammtype_1.0',
                    'mammtype_2.0'
                    ]

model = None


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
    h,w,_ = img.shape
    img = cv2.resize(img, (shape, shape))
    imgs.append(img)
    model = keras.models.load_model('model/keras_biopsy.h5')
    preds = model.predict(np.array(imgs))
    output = cv2.resize(preds[15][:,:,0], (w,h))
    #plt.imsave('predicted.png',preds[15][:,:,0])
    plt.imsave('predicted.png',output)
    with open("predicted.png", "rb") as f:
        image_binary = f.read()

    response = make_response(base64.b64encode(image_binary))
    response.headers.set('Content-Type', 'image/png')
    response.headers.set('Content-Disposition', 'attachment', filename='mask.png')
    return response


@app.route('/mammogram',methods=['POST'])
def predict_mammogram():
    data = request.get_json()

    '''
        {
            'age' : float,
            'density':int,
            'famhx' : int,
            'hrt' : int, 
            'prvmam': int, 
            'biophx' : int,
            'mammtype': int, 
            'bmi':float
        }

    '''

    data['age'] = data['age']/100.0
    data['bmi'] = data['bmi']/100.0

    ar = np.array([[data['age'], 
                    data['bmi'], 
                    1 if data['density']==1 else 0,
                    1 if data['density']==2 else 0,
                    1 if data['density']==3 else 0,
                    1 if data['density']==4 else 0,
                    1 if data['famhx']==0 else 0,
                    1 if data['famhx']==1 else 0,
                    1 if data['famhx']==9 else 0, 
                    1 if data['hrt']==0 else 0,
                    1 if data['hrt']==1 else 0,
                    1 if data['hrt']==9 else 0, 
                    1 if data['prvmam']==0 else 0,
                    1 if data['prvmam']==1 else 0,
                    1 if data['prvmam']==9 else 0, 
                    1 if data['biophx']==0 else 0,
                    1 if data['biophx']==1 else 0,
                    1 if data['biophx']==9 else 0, 
                    1 if data['mammtype']==1 else 0,
                    1 if data['mammtype']==2 else 0
                    ]])

    df2 = pd.DataFrame(ar, columns = final_df_columns)

    val = model.predict_proba(df2)
    
    return jsonify({"assess_1": val[0][0],
                    "assess_2": val[0][1],
                    "assess_3": val[0][2],
                    "assess_4": val[0][3],
                    "assess_5": val[0][4],
                    "assess_6": val[0][5],
                    })
    '''else:
        return jsonify({"error":"Bad Request." , "Description":"Bad Method. Only POST is accepted"})
    '''

# A welcome message to test our server
@app.route('/')
def index():
    return render_template('index.html')

#@app.before_first_request
#def init_model():

if __name__ == '__main__':
    model =  load('model/mammography.joblib')
    app.run(debug=True, use_reloader=False)
