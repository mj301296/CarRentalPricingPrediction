import car_price_predictor
from pathlib import Path
import yaml

# Load the config file

PACKAGE_ROOT = Path(car_price_predictor.__file__).resolve().parent
CONFIG_PATH = PACKAGE_ROOT/"config.yaml"
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)
    
DATA_PATH = PACKAGE_ROOT/config['data_path']
MODEL_PATH = PACKAGE_ROOT/config['model_save_path']
ENCODERS_PATH = PACKAGE_ROOT/config['encoders_path']
SCALAR_PATH = PACKAGE_ROOT/config['scaler_path']
RANDOM_STATE = config['random_state']
TEST_SIZE = config['test_size']
MAX_DEPTH = config['max_depth']
