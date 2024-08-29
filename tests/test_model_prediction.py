# tests/test_model_prediction.py
import unittest
import pandas as pd
from car_price_predictor.scripts.model_prediction import load_model,load_encoders_and_scaler, preprocess_new_data, predict_price


class TestModelPrediction(unittest.TestCase):

    def setUp(self):
        # Sample new data for prediction
        self.new_data = {
            "year": 2015,
            "make": "Kia",
            "trim": "LX",
            "body": "SUV",
            "condition": 5,
            "odometer": 16639.0,
            "transmission": "automatic",
        }
        
        self.new_data2 = {
            "year": 2014,
            "make": "Hyundai",
            "trim": "SE",
            "body": "Sedan",
            "condition": 5,
            "odometer": 10000.0,
            "transmission": "automatic",
        }
        # Load encoders, scaler, and model
        self.encoders, self.scaler = load_encoders_and_scaler()
        self.model = load_model()

    def test_preprocess_new_data(self):
        processed_data = preprocess_new_data(self.new_data, self.encoders, self.scaler)
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertEqual(processed_data.shape[1], len(self.new_data))
    
    def test_predict_price(self):
        processed_data = preprocess_new_data(self.new_data, self.encoders, self.scaler)
        predicted_price = predict_price(self.model, processed_data)
        self.assertEqual(predicted_price.shape[1], 5)  # Check if we predict all five output variables
    
    def test_predict_price2(self):
        processed_data = preprocess_new_data(self.new_data2, self.encoders, self.scaler)
        predicted_price = predict_price(self.model, processed_data)
        self.assertEqual(predicted_price.shape[1], 5)  # Check if we predict all five output variables

if __name__ == "__main__":
    unittest.main()
