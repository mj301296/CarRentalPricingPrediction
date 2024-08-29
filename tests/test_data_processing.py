# tests/test_data_processing.py
import unittest
import pandas as pd
from car_price_predictor.scripts.data_processing import encode_features, scale_features

class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.data = pd.read_csv('car_price_predictor/dataset/car_prices.csv')
        self.data.dropna(axis=0, inplace=True)
        self.data = self.data[["year", "make", "trim", "body", "condition", "odometer", "transmission"]]
    
    def test_load_data(self):
        self.assertIsInstance(self.data, pd.DataFrame)
        self.assertFalse(self.data.isnull().values.any())
    
    def test_encode_features(self):
        encoded_data, label_encoders = encode_features(self.data)
        self.assertIsInstance(encoded_data, pd.DataFrame)
        for column in ["year", "make", "trim", "body", "condition", "odometer", "transmission"]:
            self.assertIn(column, label_encoders)
            self.assertTrue(all(isinstance(val, int) for val in encoded_data[column]))
    
    def test_scale_features(self):
        encoded_data, _ = encode_features(self.data)
        scaled_data, _ = scale_features(encoded_data)
        self.assertIsInstance(scaled_data, pd.DataFrame)
        self.assertTrue((scaled_data.mean().abs() < 1e-6).all())  # Check if data is centered around 0
        self.assertTrue((scaled_data.std().round(5) == 1).all())   # Check if data is scaled

if __name__ == "__main__":
    unittest.main()
