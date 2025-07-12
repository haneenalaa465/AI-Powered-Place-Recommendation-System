import os
from dotenv import load_dotenv

load_dotenv() 

MAPS_API_KEY = os.getenv("MAPS_API_KEY")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

# upload Basir dataset to the raw data dir
BASIR_INPUT_CSV_PATH = os.path.join(RAW_DATA_DIR, 'Basir Research dataset - Restaurants .csv')
