import sys

import json
import numpy as np
import pandas as pd

import pickle


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


def train():

    print('Running training')

    insurance_data = pd.read_csv("insurance.csv")
    print('Dataframe dimensions:', insurance_data.shape)

    X = insurance_data.drop(columns = 'charges')
    y = insurance_data['charges']
    X_train, X_test, y_train, y_test = train_test_split(
        X , y, test_size = 0.3, random_state = 144)

    rf_model = RandomForestRegressor()

    categorical_features = ['sex', 'smoker', 'region']
        
    categorical_transformer = Pipeline(
        steps = [('encoder_cat', OneHotEncoder(
                    handle_unknown = 'ignore', drop='first',sparse_output=False))
        ]
    )

    preprocessor=ColumnTransformer(
        transformers = [('cat', categorical_transformer, categorical_features)], 
        remainder = StandardScaler()
    )

    pipe_rf = Pipeline(
        steps = [('preprocessor', preprocessor), ('regressor', rf_model)]
    )

    pipe_rf.fit(X_train, y_train)
    
    y_pred = pipe_rf.predict(X_test)
        
    training_score = pipe_rf.score(X_train, y_train)
    mean_abs_error = mean_absolute_error(y_test, y_pred)
    root_mean_sq_error = mean_squared_error(y_test, y_pred, squared = False)
    r_sq_score = r2_score(y_test, y_pred)

    with open('model.pkl', 'wb') as files:
        pickle.dump(pipe_rf, files)

    print('Training score:', training_score)
    print('MAE:', mean_abs_error)
    print('RMSE:', root_mean_sq_error)
    print('Test R2_score:',r_sq_score)

    with open("metrics.json", 'w') as outfile:
        json.dump({
            'Training score': training_score, 
            'MAE': mean_abs_error, 
            'RMSE': root_mean_sq_error, 
            'Test R2_score':r_sq_score
        }, outfile)



def predict():

    print('Running predictions')

    insurance_data = pd.read_csv("insurance.csv")

    X = insurance_data.drop(columns = 'charges')
    y = insurance_data['charges']
    X_train, X_test, y_train, y_test = train_test_split(
        X , y, test_size = 0.3, random_state = 143)


    # Load the model from the pickle file
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)


    y_pred = model.predict(X_test)
        
    mean_abs_error = mean_absolute_error(y_test, y_pred)
    root_mean_sq_error = mean_squared_error(y_test, y_pred, squared = False)
    r_sq_score = r2_score(y_test, y_pred)

    print('MAE:', mean_abs_error)
    print('RMSE:', root_mean_sq_error)
    print('Test R2_score:',r_sq_score)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Please provide an argument: 'train' or 'predict'")
    else:
        argument = sys.argv[1]
        
        if argument == "train":
            train()
        elif argument == "predict":
            predict()
        else:
            print("Invalid argument. Please provide 'train' or 'predict'")
