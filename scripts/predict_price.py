from car_price_predictor.model_prediction import load_encoders_and_scaler, preprocess_new_data, predict_price
import joblib

# Load model, encoders, and scaler
model = joblib.load("../car_price_predictor/car_rent_predictor.joblib")
encoders, scaler = load_encoders_and_scaler()

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
