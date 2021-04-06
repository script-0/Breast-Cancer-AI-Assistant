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

from imblearn.over_sampling import SMOTENC

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier


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
    global min_age
    global max_age 
    global max_bmi
    global min_bmi

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

    #update min and max age 
    if (data['age'] > max_age):
        max_age = data['age']
        print('Updating max age')
    elif (data['age'] < min_age):
        # global min_age
        min_age = data['age']
        print('Updating min age')
    else:
        pass

    #update min and max bmi
    if (data['bmi'] > max_bmi):
        #global max_age
        print('Updating max bmi from',max_age)
        max_bmi = data['bmi']
    elif (data['bmi'] < min_bmi):
        #global min_bmi
        min_bmi = data['bmi']
        print('Updating min bmi')
    else:
        pass

    #normalize age
    #data['age'] = (data['age'] - min_age)/( max_age - min_age)
    data['age'] = data['age']/100.0

    #normalize bmi
    #data['bmi'] = (data['bmi'] - min_bmi)/( max_bmi - min_bmi)
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
    """# Import dataset"""
    df = pd.read_csv(DATA_PATH)
    
    """# Preprocessing"""
    
    #Remove all row with missing bmi_c (bmi_c == -99)
    df = df.drop(df[df['bmi_c']==-99.0].index, )

    #Duplicate sample with assess_c == 5 
    df= df.append(df[df['assess_c']==5].iloc[0:2,] )
    df= df.append(df[df['assess_c']==5].iloc[0:2,] )

    #Split feature to target
    X = df.drop(['assess_c'],axis=1)
    y = df['assess_c']

    #Remove irrelevant columns
    cols_to_drop = {'cancer_c','compfilm_c','CaTypeO','ptid'}
    X.drop(columns=cols_to_drop, inplace=True)

    # Numerical features
    numerical_features = ['age_c','bmi_c']

    # Categorical features for X
    categorical_features = [col for col in X.columns if (col not in numerical_features)]

    # Categorical features index
    categorical_features_index = [X.columns.to_list().index(categorical_features[i]) for i in range(len(categorical_features)) ]

    # Dataset is very unbalanced : use [SMOTE] with categorical variables
    oversample = SMOTENC(categorical_features=categorical_features_index)
    col_tmp = X.columns
    X, y = oversample.fit_resample(X, y)

    X = pd.DataFrame(X, columns=col_tmp)

    for feature in categorical_features:
        X[feature] = X[feature].astype('category')

    #Save min and max of numerical variables
    #global min_age
    min_age = X['age_c'].min()
    
    #global max_age 
    max_age = X['age_c'].max()

    #global min_bmi
    min_bmi = X['bmi_c'].max()

    #global max_bmi
    max_bmi = X['bmi_c'].min() 

    # Preprocessing for numerical features : Normalize same as Standard Scaler
    for feature in numerical_features:
        #X[feature] = (X[feature] - np.min(X[feature])) / (np.max(X[feature]) - np.min(X[feature]))
	X[feature] = X[feature] / 100.0

    # Preprocessing for categorical features : same as OneHotEncoder
    X = pd.get_dummies(X)

    # Split training samples and test samples
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=0)
    
    """## Random Forest"""
    #global model
    model = RandomForestClassifier(n_estimators=1000, random_state=1)
   
    #global model
    model.fit(x_train,y_train)
    #model.fit(X,y)
    app.run(debug=True, use_reloader=False)
