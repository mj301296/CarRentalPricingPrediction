import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
from car_price_predictor.config.core import DATA_PATH, ENCODERS_PATH, SCALAR_PATH

def load_data():
    dataset = pd.read_csv(DATA_PATH)
    dataset.dropna(axis=0, inplace=True)
    return dataset

def encode_features(dataset):
    label_encoders = {}
    for column in ["year", "make", "trim", "body", "condition", "odometer", "transmission"]:
        le = LabelEncoder()
        dataset[column] = le.fit_transform(dataset[column])
        label_encoders[column] = le
    return dataset, label_encoders

def scale_features(input_data):
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(input_data), columns=input_data.columns)
    return scaled_data, scaler

def save_encoders_and_scaler(label_encoders, scaler):
    for column, le in label_encoders.items():
        with open(f"{ENCODERS_PATH}/{column}_le.pkl", 'wb') as file:
            pickle.dump(le, file)
    with open(f"{SCALAR_PATH}/scaler.pkl", 'wb') as file:
        pickle.dump(scaler, file)
