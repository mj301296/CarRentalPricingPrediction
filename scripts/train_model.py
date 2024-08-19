from car_price_predictor.data_processing import load_data, encode_features, scale_features, save_encoders_and_scaler
from car_price_predictor.model_training import split_data, train_model, evaluate_model, save_model

# Load and process data
dataset = load_data("../data/car_prices.csv")
dataset, label_encoders = encode_features(dataset)
input_data = dataset[["year", "make", "trim", "body", "condition", "odometer", "transmission"]]
output_data = dataset[["sellingprice", "hourly", "daily", "weekly", "monthly"]]
input_data, scaler = scale_features(input_data)

# Split data and train model
x_train, x_test, y_train, y_test = split_data(input_data, output_data)
rf = train_model(x_train, y_train)

# Evaluate and save model
evaluate_model(rf, x_train, y_train, x_test, y_test)
save_model(rf)
save_encoders_and_scaler(label_encoders, scaler)
