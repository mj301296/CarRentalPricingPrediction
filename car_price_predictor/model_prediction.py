import pandas as pd
import pickle
import yaml

# Load the config file
with open("car_price_predictor/config.yaml", 'r') as file:
    config = yaml.safe_load(file)
    
data_path = config['data_path']
model_save_path = config['model_save_path']
encoders_path = config['encoders_path']
scalar_path = config['scaler_path']


def load_model():
    with open(model_save_path, 'rb') as file:
        return pickle.load(file)

def load_encoders_and_scaler():
    encoders = {}
    for column in ["year", "make", "trim", "body", "condition", "transmission"]:
        with open(f"{encoders_path}{column}_le.pkl", 'rb') as file:
            encoders[column] = pickle.load(file)
    with open(f"{scalar_path}scaler.pkl", 'rb') as file:
        scaler = pickle.load(file)
    return encoders, scaler

def preprocess_new_data(new_data, encoders, scaler):
    for column, le in encoders.items():
        new_data[column] = le.transform([new_data[column]])[0]
    new_data_df = pd.DataFrame([new_data])
    new_data_standardized = pd.DataFrame(scaler.transform(new_data_df), columns=new_data_df.columns)
    return new_data_standardized

def predict_price(model, new_data_standardized):
    return model.predict(new_data_standardized)
