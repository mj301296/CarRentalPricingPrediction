import pandas as pd
import pickle
from car_price_predictor.config.core import MODEL_PATH, ENCODERS_PATH, SCALAR_PATH

def load_model():
    with open(f"{MODEL_PATH}/car_rent_predictor.pkl", 'rb') as file:
        return pickle.load(file)

def load_encoders_and_scaler():
    encoders = {}
    for column in ["year", "make", "trim", "body", "condition", "transmission"]:
        with open(f"{ENCODERS_PATH}/{column}_le.pkl", 'rb') as file:
            encoders[column] = pickle.load(file)
    with open(f"{SCALAR_PATH}/scaler.pkl", 'rb') as file:
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
