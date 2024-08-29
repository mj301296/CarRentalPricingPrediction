# CarRentalPricingPrediction

- This package predicts car rental prices based on various features like car model, type, odometer reading, model year, condition, transmission
- The model is trained using Ramdom Forest Regressor Classifier
- It uses sklearn's preprocessing function like LabelEncoder and StandardScaler to transform features into machine-learning-compatible formats, ensuring model accuracy and robustness
- Wheel and setuptools is used to build the python package.

### Explanation of Files and Directories

- **`analysis/`**:

  - **`data_exploration.ipynb`**:Contains Jupyter notebooks for data exploration and analysis.

- **`MANIFEST.in`**: Specifies additional files to include in the package distribution that are not automatically included by `setuptools`.

- **`Makefile`**: Provides instructions for automating tasks such as building, testing, and cleaning the project. It simplifies common commands into a single file.

- **`README.md`**: Contains information about the project, including setup instructions, usage, and other relevant details.

- **`car_price_predictor/`**: Contains the core code for the project.

  - **`config/`**: Contains configuration-related files.
    - **`core.py`**: Contains core configuration and utility functions to setup appropriate path for the files mentioned in config.yaml.
  - **`dataset/`**: Directory for storing dataset files.
    - **`car_prices.csv`**: CSV file containing car price data.
  - **`predict_price.py`**: Script for making predictions with the trained model.
  - **`scripts/`**: Contains scripts for various tasks related to data processing and model training.
    - **`data_processing.py`**: Contains functions for preprocessing data.
    - **`model_prediction.py`**: Contains functions for loading the model and making predictions.
    - **`model_training.py`**: Contains functions for training the machine learning model.
  - **`trained/`**: Directory for storing trained models and encoders.
    - **`encoders/`**: Contains saved label encoders.
      - **`body_le.pkl`**: Label encoder for `body` feature.
      - **`condition_le.pkl`**: Label encoder for `condition` feature.
      - **`make_le.pkl`**: Label encoder for `make` feature.
      - **`odometer_le.pkl`**: Label encoder for `odometer` feature.
      - **`transmission_le.pkl`**: Label encoder for `transmission` feature.
      - **`trim_le.pkl`**: Label encoder for `trim` feature.
      - **`year_le.pkl`**: Label encoder for `year` feature.
    - **`model/`**: Contains the saved machine learning model.
      - **`car_rent_predictor.pkl`**: Pickled Random Forest model for predicting car rental prices.
    - **`scaler/`**: Contains the saved scaler used for feature scaling.
      - **`scaler.pkl`**: Pickled scaler object.
  - **`config.yaml`**: Configuration file used by the application and scripts.

- **`requirements.txt`**: Lists the Python dependencies required for the project.

- **`setup.py`**: Contains metadata and instructions for building and installing the package.

- **`tests/`**: Contains unit tests for the project.

  - **`test_data_processing.py`**: Tests for the data processing functions.
  - **`test_model_prediction.py`**: Tests for the model prediction functions.
  - **`test_model_training.py`**: Tests for the model training functions.

- **`train_model.py`**: Script for training the machine learning model. It preprocesses data, trains the model, and saves the model and encoders to the `trained` directory.

## Installation of package locally

pip install .

## Usage

1. Train the model: Generate .pkl files for model, encoders and scaler

python train_model.py

2. Predict prices: Utilizes the generated .pkl files to predict prices

python car_rent_predictor/predict_price.py

3. Build package: Generates a build/ and dist/ which includes a .whl and .tar.gz package files
   python setup.py sdist bdist_wheel

4. Uploading package to Amazon S3 bucket:
   aws s3 cp dist/ s3://rental-price-predictor-package/car_rent_predictor-v%number%/ --recursive

# FastApi Application: a FastAPI microservice to deliver an API for model predictions, ensuring fast response times and low latency

https://github.com/mj301296/CarRentalPredictionApi
