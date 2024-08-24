from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
from car_price_predictor.config.core import MODEL_PATH, RANDOM_STATE, TEST_SIZE, MAX_DEPTH

def split_data(input_data, output_data):
    x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return x_train, x_test, y_train, y_test

def train_model(x_train, y_train):
    rf = RandomForestRegressor(random_state=RANDOM_STATE, max_depth=MAX_DEPTH)
    rf.fit(x_train, y_train)
    return rf

def evaluate_model(rf, x_train, y_train, x_test, y_test):
    y_train_pred = rf.predict(x_train)
    y_test_pred = rf.predict(x_test)
    train_rmse_rf = mean_squared_error(y_train, y_train_pred, squared=False)
    test_rmse_rf = mean_squared_error(y_test, y_test_pred, squared=False)
    print(f"Train RMSE: {train_rmse_rf}")
    print(f"Test RMSE: {test_rmse_rf}")
    print("R squared for train is :", rf.score(x_train, y_train) * 100)
    print("R squared for test is :", rf.score(x_test, y_test) * 100)

def save_model(rf):
    with open(f"{MODEL_PATH}/car_rent_predictor.pkl", 'wb') as file:
        pickle.dump(rf, file)
