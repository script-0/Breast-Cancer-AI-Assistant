from flask import Flask, request, jsonify
from flask_cors import CORS

#Model imports
import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTENC

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

#Global variables
seed=11
DATA_PATH = "dataset/breast_mammogram_dataset.csv"

max_age=0
min_age=0 
max_bmi=0
min_bmi=0

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

app = Flask(__name__)
CORS(app)

@app.route('/mammogram',methods=['POST'])
def predict_mammogram():
    if request.method == 'POST':
        val = 50
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
            min_age = data['age']
            print('Updating min age')
        else:
            
        #update min and max bmi
        if (data['bmi'] > max_bmi):
            max_bmi = data['bmi']
            print('Updating max bmi')
        elif (data['bmi'] < min_bmi):
            min_age = data['bmi']
            print('Updating min bmi')
        else:

        #normalize age
        data['age'] = (data['age'] - min_age)/(max_age-min_age)

        #normalize bmi
        data['bmi'] = (data['bmi'] - min_bmi)/(max_bmi-min_bmi)

        
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
    else:
        return jsonify({"error":"Bad Request." , "Description":"Bad Method. Only POST is accepted"})

# A welcome message to test our server
@app.route('/')
def index():
    return "<h1>Welcome to our Breast Cancer Assistant API !!</h1>"

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
    min_age , max_age = X['age_c'].min() , X['age_c'].max()
    min_bmi , max_bmi = X['bmi_c'].min() , X['bmi_c'].max()

    # Preprocessing for numerical features : Normalize same as Standard Scaler
    for feature in numerical_features:
        X[feature] = (X[feature] - np.min(X[feature])) / (np.max(X[feature]) - np.min(X[feature]))

    # Preprocessing for categorical features : same as OneHotEncoder
    X = pd.get_dummies(X)

    # Split training samples and test samples
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=0)
    
    """## Random Forest"""
    model = RandomForestClassifier(n_estimators=1000, random_state=1)
    model.fit(x_train,y_train)

    app.run(threaded=True, port=5002)
