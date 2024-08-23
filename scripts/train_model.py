from car_price_predictor.data_processing import load_data, encode_features, scale_features, save_encoders_and_scaler
from car_price_predictor.model_training import split_data, train_model, evaluate_model, save_model
import yaml

# Load the config file
with open("car_price_predictor/config.yaml", 'r') as file:
    config = yaml.safe_load(file)
    
data_path = config['data_path']
model_save_path = config['model_save_path']
encoders_path = config['encoders_path']
scalar_path = config['scaler_path']
random_state = config['random_state']
test_size = config['test_size']
max_depth = config['max_depth']
# Load and process data
print("Loading dataset...")
dataset = load_data(data_path)
print("Pre-processing data...")
print("Encoding features...")
dataset, label_encoders = encode_features(dataset)
input_data = dataset[["year", "make", "trim", "body", "condition", "odometer", "transmission"]]
output_data = dataset[["sellingprice", "hourly", "daily", "weekly", "monthly"]]
print("Scaling features...")
input_data, scaler = scale_features(input_data)

# Split data and train model
print("Spliting the dataset...")
x_train, x_test, y_train, y_test = split_data(input_data, output_data, test_size, random_state)
print("Training the model...")
rf = train_model(x_train, y_train, random_state, max_depth)

# Evaluate and save model
print("Evaluate model...")
evaluate_model(rf, x_train, y_train, x_test, y_test)
print("Saving model..")
save_model(rf, model_save_path)
print("Saving encoders and scaler...")
save_encoders_and_scaler(label_encoders, scaler, encoders_path, scalar_path)
