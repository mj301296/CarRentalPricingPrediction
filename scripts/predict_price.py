from car_price_predictor.model_prediction import load_model,load_encoders_and_scaler, preprocess_new_data, predict_price
import joblib
import yaml

# Load the config file
with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)
    
model_save_path = config['model_save_path']
encoders_path = config['encoders_path']
scalar_path = config['scaler_path']

# Load model, encoders, and scaler
model = load_model(model_save_path)
encoders, scaler = load_encoders_and_scaler(encoders_path, scalar_path)

# New data
new_data = {
    "year": 2015,
    "make": "Kia",
    "trim": "LX",
    "body": "SUV",
    "condition": 5,
    "odometer": 16639.0,
    "transmission": "automatic",
}

# Process and predict
new_data_standardized = preprocess_new_data(new_data, encoders, scaler)
predicted_price = predict_price(model, new_data_standardized)

# Print the predicted price
print("Predicted Selling Price:", predicted_price)
